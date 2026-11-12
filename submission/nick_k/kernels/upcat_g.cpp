// Custom kernel (team Nick): fused "nearest-2x upsample + channel concat" for the YOLOv4 neck, generic version
// for bf16 TILE tensors and an L1-INTERLEAVED output (the 384/512/640 graphs). Pure data movement; the same
// source runs on BRISC (NoC0) and NCRISC (NoC1), each with its own host-built work list.
//
// bf16 tile (2048 B) = 4 faces x 16 rows x 16 values; one tile row (= one pixel's 32 channels) is two 32 B face
// rows: face (r / 16) * 2 at row r % 16 and the face to its right (+512 B). Moving pixels = moving 32 B pieces:
// byte-exact, no untilize / tilize.
//
// Runtime args:
//   out_addr, a_addr, b_addr, scratch_slot0, n_src, n_lat, n_up,
//   n_src x [noc_xy, a_off]       whole A (low-res) tiles -> scratch slots slot0 + i
//   n_lat x [b_off, page]         B (lateral) tile in THIS core's shard -> output page (NoC write)
//   n_up  x [page, 16 words map]  output page gathered from scratch; entry (16 bit) = slot << 5 | src_row,
//                                 0xFFFF = zero row. Built in one of NB build slots, then NoC-written.
// noc_xy = x | y << 16 (virtual coordinates).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE = 2048;
constexpr uint32_t FACE = 512;
constexpr uint32_t FROW = 32;
constexpr uint32_t NB = 4;  // build slots (ring)

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    uint32_t i = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(i++);
    const uint32_t a_addr = get_arg_val<uint32_t>(i++);
    const uint32_t b_addr = get_arg_val<uint32_t>(i++);
    const uint32_t slot0 = get_arg_val<uint32_t>(i++);
    const uint32_t n_src = get_arg_val<uint32_t>(i++);
    const uint32_t n_lat = get_arg_val<uint32_t>(i++);
    const uint32_t n_up = get_arg_val<uint32_t>(i++);
    const uint32_t scratch = get_write_ptr(0) + slot0 * TILE;
    const uint32_t build0 = scratch + n_src * TILE;
    const auto out = TensorAccessor(out_args, out_addr, TILE);

    for (uint32_t s = 0; s < n_src; ++s) {
        const uint32_t xy = get_arg_val<uint32_t>(i++);
        const uint32_t off = get_arg_val<uint32_t>(i++);
        noc_async_read(get_noc_addr(xy & 0xFFFF, xy >> 16, a_addr + off), scratch + s * TILE, TILE);
    }
    for (uint32_t l = 0; l < n_lat; ++l) {
        const uint32_t off = get_arg_val<uint32_t>(i++);
        const uint32_t page = get_arg_val<uint32_t>(i++);
        noc_async_write_tile(page, out, b_addr + off);
    }
    noc_async_read_barrier();

    for (uint32_t u0 = 0; u0 < n_up; u0 += NB) {
        const uint32_t nb = n_up - u0 < NB ? n_up - u0 : NB;
        if (u0) {
            noc_async_writes_flushed();  // build slots are free again
        }
        noc_async_read_one_packet_set_state(get_noc_addr(scratch), FROW);
        uint32_t ia = i;
        for (uint32_t k = 0; k < nb; ++k, ia += 17) {
            const uint32_t dst = build0 + k * TILE;
            uint32_t prev = 0xFFFF, s_m = 0;
            for (uint32_t pr = 0; pr < 16; ++pr) {
                const uint32_t pair = get_arg_val<uint32_t>(ia + 1 + pr);
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t r = pr * 2 + h;
                    const uint32_t e = h ? (pair >> 16) : (pair & 0xFFFF);
                    const uint32_t d_m = dst + (r >> 4) * 2 * FACE + (r & 15) * FROW;
                    if (e == 0xFFFF) {
                        volatile tt_l1_ptr uint32_t* d0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d_m);
                        volatile tt_l1_ptr uint32_t* d1 = d0 + FACE / 4;
                        for (uint32_t w = 0; w < FROW / 4; ++w) {
                            d0[w] = 0;
                            d1[w] = 0;
                        }
                        continue;
                    }
                    if (e != prev) {
                        prev = e;
                        const uint32_t j = e & 31;
                        s_m = scratch + (e >> 5) * TILE + (j >> 4) * 2 * FACE + (j & 15) * FROW;
                    }
                    noc_async_read_one_packet_with_state(s_m, d_m);
                    noc_async_read_one_packet_with_state(s_m + FACE, d_m + FACE);
                }
            }
        }
        noc_async_read_barrier();
        ia = i;
        for (uint32_t k = 0; k < nb; ++k, ia += 17) {
            noc_async_write_tile(get_arg_val<uint32_t>(ia), out, build0 + k * TILE);
        }
        i = ia;
    }
    noc_async_write_barrier();
}
