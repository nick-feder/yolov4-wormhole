// Custom kernel (team Nick): SPP writer. Compute packs 12 tiles: i = part * 4 + tile_row with
// part 0 = max5, 1 = max9, 2 = max13; the concat order is [max13, max9, max5, x] (16 tile columns each).
// Runtime args: out_addr, j (this core's tile column within a part).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_out = 16, n_rows = 4, n_parts = 3, n_out = n_rows * n_parts;
    constexpr uint32_t out_tiles_per_row = 64;  // 2048 output channels / 32
    constexpr uint32_t part_cols = 16;          // 512 channels per part / 32

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t j = get_arg_val<uint32_t>(1);
    const uint32_t tile = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, tile);

    cb_wait_front(cb_out, n_out);
    const uint32_t l1 = get_read_ptr(cb_out);
    for (uint32_t i = 0; i < n_out; ++i) {
        const uint32_t part = i / n_rows, row = i % n_rows;
        const uint32_t col = (n_parts - 1 - part) * part_cols + j;
        noc_async_write_tile(row * out_tiles_per_row + col, out, l1 + i * tile);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, n_out);
}
