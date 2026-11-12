// Custom kernel (team Nick): fused "nearest-2x upsample + channel concat" for the YOLOv4 neck.
// Pure data movement, runs on both RISC-V data-movement cores (BRISC on NoC0, NCRISC on NoC1),
// each with its own work list from the host.
//
// Everything stays in bfp8_b TILE format. In a bfp8_b tile one shared exponent covers the 16 values
// of one face row (= 16 channels of one pixel), so moving whole tile rows (pixels) between tiles is
// byte-exact: copy 1 exponent byte + 16 mantissa bytes per face. No untilize / tilize / requantize.
//
// Tile layout (1088 B): [exponents 4 faces x 16 rows = 64 B][mantissas 4 faces x 16 rows x 16 B].
// Faces: f0 = rows 0-15 / cols 0-15, f1 = rows 0-15 / cols 16-31, f2 = rows 16-31 / cols 0-15, f3.
//
// Runtime args:
//   out_addr, a_addr, b_addr, scratch_slot0, n_src, n_lat, n_up,
//   n_src x [noc_xy, src_off]                  whole A tiles -> scratch slots (slot0 + i)
//   n_lat x [noc_xy, src_off, dst_off]         whole B tiles -> output (lateral half of the concat)
//   n_up  x [dst_xy, dst_off, 16 words map]    output tile gathered from scratch; entry (16 bit) =
//                                              slot << 5 | src_row, 0xFFFF = zero row.
//                                              dst_xy == 0: tile is in this core's output shard;
//                                              else it is built in the next free scratch slot and
//                                              NoC-written to core dst_xy (load balancing: the up
//                                              half of the channels is owned by half of the cores).
// noc_xy = x | y << 16 (virtual coordinates; never 0 for a worker core).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
// v3: like upcat_dm.cpp (v2), but mantissa rows are moved by local NoC reads instead of RISC-V loads/stores.

constexpr uint32_t TILE = 1088;
constexpr uint32_t EXP = 64;

void kernel_main() {
    uint32_t i = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(i++);
    const uint32_t a_addr = get_arg_val<uint32_t>(i++);
    const uint32_t b_addr = get_arg_val<uint32_t>(i++);
    const uint32_t slot0 = get_arg_val<uint32_t>(i++);
    const uint32_t n_src = get_arg_val<uint32_t>(i++);
    const uint32_t n_lat = get_arg_val<uint32_t>(i++);
    const uint32_t n_up = get_arg_val<uint32_t>(i++);
    const uint32_t scratch = get_write_ptr(0) + slot0 * TILE;
    uint32_t build = scratch + n_src * TILE;

    for (uint32_t s = 0; s < n_src; ++s) {
        const uint32_t xy = get_arg_val<uint32_t>(i++);
        const uint32_t off = get_arg_val<uint32_t>(i++);
        noc_async_read(get_noc_addr(xy & 0xFFFF, xy >> 16, a_addr + off), scratch + s * TILE, TILE);
    }
    for (uint32_t l = 0; l < n_lat; ++l) {
        const uint32_t xy = get_arg_val<uint32_t>(i++);
        const uint32_t off = get_arg_val<uint32_t>(i++);
        const uint32_t dst = get_arg_val<uint32_t>(i++);
        noc_async_read(get_noc_addr(xy & 0xFFFF, xy >> 16, b_addr + off), out_addr + dst, TILE);
    }
    noc_async_read_barrier();

    // Mantissa rows (16 B per face row, 16 B aligned) move as local NoC reads issued with a preset state
    // (a few register writes each, executed asynchronously); RISC-V only assembles the exponent words.
    noc_async_read_one_packet_set_state(get_noc_addr(scratch), 16);
    const uint32_t build0 = build;
    for (uint32_t u = 0; u < n_up; ++u) {
        const uint32_t dxy = get_arg_val<uint32_t>(i++);
        const uint32_t doff = get_arg_val<uint32_t>(i++);
        const uint32_t dst = dxy ? build : out_addr + doff;
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
        uint32_t ex[16];
        for (uint32_t w = 0; w < 16; ++w) {
            ex[w] = 0;
        }
        uint32_t prev = 0xFFFF, e0 = 0, e1 = 0, s_m = 0;
        for (uint32_t pr = 0; pr < 16; ++pr) {
            const uint32_t pair = get_arg_val<uint32_t>(i++);
            for (uint32_t k = 0; k < 2; ++k) {
                const uint32_t r = pr * 2 + k;
                const uint32_t e = k ? (pair >> 16) : (pair & 0xFFFF);
                const uint32_t of = (r >> 4) << 1;  // first output face of this row
                const uint32_t ri = r & 15;
                const uint32_t d_m = dst + EXP + of * 256 + ri * 16;
                if (e == 0xFFFF) {
                    volatile tt_l1_ptr uint32_t* d0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d_m);
                    volatile tt_l1_ptr uint32_t* d1 = d0 + 64;  // next face (+256 B)
                    d0[0] = 0; d0[1] = 0; d0[2] = 0; d0[3] = 0;
                    d1[0] = 0; d1[1] = 0; d1[2] = 0; d1[3] = 0;
                    continue;
                }
                if (e != prev) {
                    prev = e;
                    const uint32_t src = scratch + (e >> 5) * TILE;
                    const uint32_t j = e & 31;
                    const uint32_t sf = (j >> 4) << 1;
                    const uint32_t rj = j & 15;
                    s_m = src + EXP + sf * 256 + rj * 16;
                    volatile tt_l1_ptr uint8_t* se = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src);
                    e0 = se[sf * 16 + rj];
                    e1 = se[(sf + 1) * 16 + rj];
                }
                noc_async_read_one_packet_with_state(s_m, d_m);
                noc_async_read_one_packet_with_state(s_m + 256, d_m + 256);
                const uint32_t sh = (ri & 3) * 8;
                ex[(of * 16 + ri) >> 2] |= e0 << sh;
                ex[((of + 1) * 16 + ri) >> 2] |= e1 << sh;
            }
        }
        for (uint32_t w = 0; w < 16; ++w) {
            o[w] = ex[w];
        }
        if (dxy) {
            build += TILE;
        }
    }
    noc_async_read_barrier();
    // built tiles for other cores: send after their mantissa reads have landed
    if (build != build0) {
        uint32_t ia = 7 + 2 * n_src + 3 * n_lat;  // first n_up record
        uint32_t b = build0;
        for (uint32_t u = 0; u < n_up; ++u, ia += 18) {
            const uint32_t dxy = get_arg_val<uint32_t>(ia);
            if (dxy) {
                noc_async_write(b, get_noc_addr(dxy & 0xFFFF, dxy >> 16, out_addr + get_arg_val<uint32_t>(ia + 1)), TILE);
                b += TILE;
            }
        }
        noc_async_write_barrier();
    }
}