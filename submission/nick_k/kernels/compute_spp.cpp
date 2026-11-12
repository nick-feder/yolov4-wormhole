// Custom kernel (team Nick): YOLOv4 neck SPP block at 10x10, one core per 32 channels.
//   in  (CB 0):  4 tiles = positions 0..127 (rows, 100 valid = 10x10 map) x 32 channels of conv3's output
//   out (CB 16): 12 tiles = max5 (DST tiles 0-3), max9 (4-7), max13 (8-11), packed to the CB format (bfp8)
// Stride-1 "same" max pooling with -inf padding == clipped windows, and it cascades exactly:
//   max9 = max5(max5(x)), max13 = max5(max9(x)); each max5 is separable (1x5 along w, then 5x1 along h).
// Max is exact, so the result equals torch max_pool2d on the bf16 values bit for bit.
//
// All work is done on the SFPU inside one DST acquire (dst_full_sync_en: 16 bf16 tiles = 1024 DST rows).
// SFPU registers hold 32 lanes = 4 tile rows x 8 columns, so pooling across rows (= positions) needs the
// data "position-major": group of 4 positions -> SFPTRANSP -> LREG k holds all 32 channels of one position,
// stored to slot C + 2p (one slot = one SFPLOAD/SFPSTORE address, even and odd column halves interleave).
//   DST rows   0- 255  S5 : input tiles, later max5 (standard tile layout)
//            256- 511  S9 : max9
//            512- 767  S13: max13; rows 512-711 double as T (h-pass temp, dead before S13 is written)
//            768- 967  C  : position-major slots, position p = 10h + w at row 768 + 2p
// Rows 100-127 (tile padding) of S5 keep the input, of S9/S13 whatever DST held; they only affect padding.
// SPP_MODE 1: layout round trip only (all three outputs == input), to validate the transposes.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
#include "ckernel_addrmod.h"
#include "llk_math_eltwise_sfpu_common.h"

namespace spp {
using namespace ckernel;

constexpr uint32_t S5 = 0, S9 = 256, S13 = 512, T = 512, C = 768;
constexpr uint32_t N = 10;       // map is N x N
constexpr uint32_t ROW = 2 * N;  // slot distance of h -> h+1
constexpr uint32_t GROUPS = N * N / 4;
constexpr uint32_t FMT = 2;  // InstrModLoadStore::FP16B (DST holds bf16)
constexpr uint32_t LR0 = 0, LR1 = 1, LR2 = 2, LR3 = 3, LR4 = 4;
// SFPU instructions address ADDR_MOD_4..7 (base set by _llk_math_eltwise_sfpu_start_)
constexpr uint32_t AM_0 = 3;  // ADDR_MOD_7: no increment
constexpr uint32_t AM_H = 1;  // ADDR_MOD_5: dest += ROW (next line h)
constexpr uint32_t AM_W = 2;  // ADDR_MOD_6: dest += 2   (next line w)

inline void config_addr_mods() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = ROW}}.set(ADDR_MOD_5);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}

inline void reset_d() { TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D); }

// Output element K of all 10 lines: max over the clipped window [K-2, K+2] of the line elements (stride E
// slots, region SRC) -> DST + E*K. The line index runs through the dest RW counter (+STEP per store).
// SFPSWAP(mod 1) leaves max in the src_c register (first operand); its 2-cycle latency is covered by
// an independent instruction or an SFPNOP before the result is read.
template <uint32_t SRC, uint32_t DST, uint32_t E, uint32_t STEP, uint32_t K>
inline void window_k() {
    constexpr uint32_t lo = K < 2 ? 0 : K - 2;
    constexpr uint32_t hi = K + 2 > N - 1 ? N - 1 : K + 2;
    constexpr uint32_t n = hi - lo + 1;
    reset_d();
#pragma GCC unroll 0
    for (uint32_t it = 0; it < N; ++it) {
        TTI_SFPLOAD(LR0, FMT, AM_0, SRC + E * lo);
        TTI_SFPLOAD(LR1, FMT, AM_0, SRC + E * (lo + 1));
        TTI_SFPLOAD(LR2, FMT, AM_0, SRC + E * (lo + 2));
        TTI_SFPSWAP(0, LR0, LR1, 1);
        if constexpr (n == 3) {
            TTI_SFPNOP;
            TTI_SFPSWAP(0, LR0, LR2, 1);
        } else if constexpr (n == 4) {
            TTI_SFPLOAD(LR3, FMT, AM_0, SRC + E * (lo + 3));
            TTI_SFPSWAP(0, LR2, LR3, 1);
            TTI_SFPNOP;
            TTI_SFPSWAP(0, LR0, LR2, 1);
        } else {
            TTI_SFPLOAD(LR3, FMT, AM_0, SRC + E * (lo + 3));
            TTI_SFPLOAD(LR4, FMT, AM_0, SRC + E * (lo + 4));
            TTI_SFPSWAP(0, LR2, LR3, 1);
            TTI_SFPSWAP(0, LR0, LR4, 1);
            TTI_SFPNOP;
            TTI_SFPSWAP(0, LR0, LR2, 1);
        }
        TTI_SFPNOP;
        TTI_SFPSTORE(LR0, FMT, STEP, DST + E * K);
    }
}

template <uint32_t SRC, uint32_t DST, uint32_t E, uint32_t STEP>
inline void line_pass() {
    window_k<SRC, DST, E, STEP, 0>();
    window_k<SRC, DST, E, STEP, 1>();
    window_k<SRC, DST, E, STEP, 2>();
    window_k<SRC, DST, E, STEP, 3>();
    window_k<SRC, DST, E, STEP, 4>();
    window_k<SRC, DST, E, STEP, 5>();
    window_k<SRC, DST, E, STEP, 6>();
    window_k<SRC, DST, E, STEP, 7>();
    window_k<SRC, DST, E, STEP, 8>();
    window_k<SRC, DST, E, STEP, 9>();
}

// One max5 stage on C (position-major): along w (C -> T, lines = rows h), then along h (T -> C, lines = w).
inline void max5_stage() {
    line_pass<C, T, 2, AM_H>();
    line_pass<T, C, ROW, AM_W>();
}

// Positions 4g..4g+3 of a standard-layout region S: tile g/8, tile rows q0..q0+3 (q0 = 4*(g%8)) live in
// face rows q0%16 of faces 0/1 (q0 < 16) or 2/3 (+32); even/odd column halves at +0/+2, right face at +16.
inline uint32_t group_row(uint32_t S, uint32_t g) {
    const uint32_t q0 = (g & 7) << 2;
    return S + ((g >> 3) << 6) + ((q0 & 16) << 1) + (q0 & 15);
}

inline void to_c(uint32_t S) {
    for (uint32_t g = 0; g < GROUPS; ++g) {
        const uint32_t r = group_row(S, g);
        const uint32_t c = C + (g << 3);
        TT_SFPLOAD(LR0, FMT, AM_0, r);
        TT_SFPLOAD(LR1, FMT, AM_0, r + 2);
        TT_SFPLOAD(LR2, FMT, AM_0, r + 16);
        TT_SFPLOAD(LR3, FMT, AM_0, r + 18);
        TTI_SFPTRANSP(0, 0, 0, 0);
        TT_SFPSTORE(LR0, FMT, AM_0, c);
        TT_SFPSTORE(LR1, FMT, AM_0, c + 2);
        TT_SFPSTORE(LR2, FMT, AM_0, c + 4);
        TT_SFPSTORE(LR3, FMT, AM_0, c + 6);
    }
}

inline void from_c(uint32_t S) {
    for (uint32_t g = 0; g < GROUPS; ++g) {
        const uint32_t r = group_row(S, g);
        const uint32_t c = C + (g << 3);
        TT_SFPLOAD(LR0, FMT, AM_0, c);
        TT_SFPLOAD(LR1, FMT, AM_0, c + 2);
        TT_SFPLOAD(LR2, FMT, AM_0, c + 4);
        TT_SFPLOAD(LR3, FMT, AM_0, c + 6);
        TTI_SFPTRANSP(0, 0, 0, 0);
        TT_SFPSTORE(LR0, FMT, AM_0, r);
        TT_SFPSTORE(LR1, FMT, AM_0, r + 2);
        TT_SFPSTORE(LR2, FMT, AM_0, r + 16);
        TT_SFPSTORE(LR3, FMT, AM_0, r + 18);
    }
}

inline void run() {
    _llk_math_eltwise_sfpu_start_(0);
    config_addr_mods();
    reset_d();
    to_c(S5);
#if SPP_MODE == 1
    from_c(S5);
    from_c(S9);
    from_c(S13);
#else
    max5_stage();
    reset_d();
    from_c(S5);
    max5_stage();
    reset_d();
    from_c(S9);
    max5_stage();
    reset_d();
    from_c(S13);
#endif
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace spp
#endif

void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_out = 16;
    constexpr uint32_t n_in = 4, n_out = 12;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_tile_init(cb_in);
    cb_wait_front(cb_in, n_in);
    tile_regs_acquire();
    for (uint32_t i = 0; i < n_in; ++i) {
        copy_tile(cb_in, i, i);
    }
    MATH((spp::run()));
    tile_regs_commit();
    cb_pop_front(cb_in, n_in);
    cb_reserve_back(cb_out, n_out);
    tile_regs_wait();
    for (uint32_t i = 0; i < n_out; ++i) {
        pack_tile(i, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, n_out);
}
