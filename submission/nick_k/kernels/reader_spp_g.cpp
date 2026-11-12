// Custom kernel (team Nick): generic SPP reader. Gathers this core's TN input tiles (one 32-channel tile column of
// conv3's output, all positions) into CB 0 for the compute kernel, then writes the same tiles to the pass-through
// part of the concat (output tile column 48 + j). Define TN.
// Runtime args: out_addr, j, TN x [noc_xy (x | y << 16), l1_addr] of the input tiles.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t out_tiles_per_row = 64;  // 2048 output channels / 32
    constexpr uint32_t x_col = 48;              // input channels sit after max13, max9, max5 (3 x 16 tiles)

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t j = get_arg_val<uint32_t>(1);
    const uint32_t tile = get_tile_size(cb_in);

    cb_reserve_back(cb_in, TN);
    const uint32_t l1 = get_write_ptr(cb_in);
    for (uint32_t t = 0; t < TN; ++t) {
        const uint32_t xy = get_arg_val<uint32_t>(2 + 2 * t);
        const uint32_t src = get_arg_val<uint32_t>(3 + 2 * t);
        noc_async_read(get_noc_addr(xy & 0xFFFF, xy >> 16, src), l1 + t * tile, tile);
    }
    noc_async_read_barrier();
    cb_push_back(cb_in, TN);

    const auto out = TensorAccessor(out_args, out_addr, tile);
    for (uint32_t t = 0; t < TN; ++t) {
        noc_async_write_tile(t * out_tiles_per_row + x_col + j, out, l1 + t * tile);
    }
    noc_async_write_barrier();
}
