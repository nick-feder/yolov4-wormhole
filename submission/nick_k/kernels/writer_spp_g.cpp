// Custom kernel (team Nick): generic SPP writer. Compute pushes 3 x TN tiles in order part (0 = max5, 1 = max9,
// 2 = max13) x tile row; the concat order is [max13, max9, max5, x] (16 tile columns each). Define TN.
// Runtime args: out_addr, j (this core's tile column within a part).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_out = 16, n_parts = 3;
    constexpr uint32_t out_tiles_per_row = 64;  // 2048 output channels / 32
    constexpr uint32_t part_cols = 16;          // 512 channels per part / 32

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t j = get_arg_val<uint32_t>(1);
    const uint32_t tile = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, tile);

    for (uint32_t part = 0; part < n_parts; ++part) {
        const uint32_t col = (n_parts - 1 - part) * part_cols + j;
        for (uint32_t t = 0; t < TN; ++t) {
            cb_wait_front(cb_out, 1);
            noc_async_write_tile(t * out_tiles_per_row + col, out, get_read_ptr(cb_out));
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
}
