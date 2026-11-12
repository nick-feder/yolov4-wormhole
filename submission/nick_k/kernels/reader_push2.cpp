// Announce this core's two L1 shards (both CBs are placed on the sharded tensor buffers): zero bytes move.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    cb_reserve_back(0, n);
    cb_push_back(0, n);
    cb_reserve_back(1, n);
    cb_push_back(1, n);
}
