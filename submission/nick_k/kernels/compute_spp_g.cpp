// Custom kernel (team Nick): YOLOv4 neck SPP block, generic map size N x N (N even, N*N <= 13*32: 384/512/640
// graphs, N = 12/16/20), one core per 32 channels, bf16. Exact (pure max), cascade max9 = max5(max5),
// max13 = max5(max9), each max5 separable (1x5 along w, then 5x1 along h), clipped windows == -inf padding.
//
// Nick's 320 kernel keeps everything in one DST acquire; that does not fit for N > 10, so the data is spilled
// through CBs between rounds. Key fact: the position-major form ("image") of positions 32t..32t+31 is exactly
// DST rows 64t..64t+63 (position p -> DST row 2p, SFPTRANSP groups of 4 positions), i.e. ONE tile slot, so the
// transposes are tile-local and an image tile can be packed / unpacked like a normal tile (bit-exact round trip).
//   A  (per batch of 8):  input tile -> to_c -> image tile                     cb_in -> cb_a
//   P  (x3, one acquire): TN image tiles -> in-place max5 (w pass, h pass)    cb_a <-> cb_b (ping-pong)
//   B  (per batch of 8):  image tile -> from_c -> standard tile               -> cb_out (writer), after each P
// The pooling passes are in place: windows are loaded from DST, the result of position K is kept in an LREG
// and stored two steps later (once position K is no longer read).
// Compile-time defines: N (map size), TN (tiles per 32-channel column = ceil(N*N/32)).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
#include "ckernel_addrmod.h"
#include "llk_math_eltwise_sfpu_common.h"

namespace sppg {
using namespace ckernel;

constexpr uint32_t ROW = 2 * N;  // DST row distance of h -> h+1
constexpr uint32_t FMT = 2;      // InstrModLoadStore::FP16B
constexpr uint32_t LR0 = 0, LR1 = 1, LR2 = 2, LR3 = 3, LR4 = 4, LP0 = 5, LP1 = 6;
constexpr uint32_t AM_0 = 3;  // ADDR_MOD_7: no increment
constexpr uint32_t AM_H = 1;  // ADDR_MOD_5: dest += ROW
constexpr uint32_t AM_W = 2;  // ADDR_MOD_6: dest += 2

inline void config_addr_mods() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = ROW}}.set(ADDR_MOD_5);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}

inline void reset_d() { TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D); }

constexpr uint32_t pend(uint32_t k) { return (k & 1) ? LP1 : LP0; }

// Step K of an in-place line pass (elements at E * k + line base in the D counter): loads the clipped window of
// K, stores the result of K-2 (kept in pend(K)), keeps the result of K in pend(K).
template <uint32_t E, uint32_t K>
inline void step() {
    constexpr uint32_t lo = K < 2 ? 0 : K - 2;
    constexpr uint32_t hi = K + 2 > N - 1 ? N - 1 : K + 2;
    constexpr uint32_t n = hi - lo + 1;
    TTI_SFPLOAD(LR0, FMT, AM_0, E * lo);
    TTI_SFPLOAD(LR1, FMT, AM_0, E * (lo + 1));
    TTI_SFPLOAD(LR2, FMT, AM_0, E * (lo + 2));
    TTI_SFPSWAP(0, LR0, LR1, 1);
    if constexpr (n == 3) {
        TTI_SFPNOP;
        TTI_SFPSWAP(0, LR0, LR2, 1);
    } else if constexpr (n == 4) {
        TTI_SFPLOAD(LR3, FMT, AM_0, E * (lo + 3));
        TTI_SFPSWAP(0, LR2, LR3, 1);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, LR0, LR2, 1);
    } else {
        TTI_SFPLOAD(LR3, FMT, AM_0, E * (lo + 3));
        TTI_SFPLOAD(LR4, FMT, AM_0, E * (lo + 4));
        TTI_SFPSWAP(0, LR2, LR3, 1);
        TTI_SFPSWAP(0, LR0, LR4, 1);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, LR0, LR2, 1);
    }
    if constexpr (K >= 2) {
        TTI_SFPSTORE(pend(K), FMT, AM_0, E * (K - 2));  // also covers the SFPSWAP latency
    } else {
        TTI_SFPNOP;
    }
    TTI_SFPMOV(0, LR0, pend(K), 0);
    if constexpr (K + 1 < N) {
        step<E, K + 1>();
    }
}

// In-place max over the clipped 5-window of N lines of N elements (element stride E, line stride via AM_LINE).
template <uint32_t E, uint32_t AM_LINE>
inline void line_pass() {
    reset_d();
#pragma GCC unroll 0
    for (uint32_t it = 0; it < N; ++it) {
        step<E, 0>();
        TTI_SFPNOP;
        TTI_SFPSTORE(pend(N - 2), FMT, AM_0, E * (N - 2));
        TTI_SFPSTORE(pend(N - 1), FMT, AM_LINE, E * (N - 1));
    }
}

inline void pool_stage() {
    _llk_math_eltwise_sfpu_start_(0);
    config_addr_mods();
    line_pass<2, AM_H>();    // along w: line h at 2N*h, element w at 2w
    line_pass<ROW, AM_W>();  // along h: line w at 2w, element h at 2N*h
    reset_d();
    _llk_math_eltwise_sfpu_done_();
}

// Positions 4gl..4gl+3 of the standard tile in slot s: face rows q0%16 of faces 0/1 (q0 < 16) or 2/3 (+32);
// even/odd column halves at +0/+2, right face at +16. Image: slot d, row 8gl + 2k for position 4gl + k.
inline void transpose_tiles(uint32_t src0, uint32_t dst0, uint32_t nt, bool to_image) {
    _llk_math_eltwise_sfpu_start_(0);
    config_addr_mods();
    reset_d();
    for (uint32_t t = 0; t < nt; ++t) {
        for (uint32_t gl = 0; gl < 8; ++gl) {
            const uint32_t q0 = gl << 2;
            const uint32_t st = (to_image ? src0 : dst0) + t;  // standard-layout slot
            const uint32_t im = (to_image ? dst0 : src0) + t;  // image slot
            const uint32_t r = (st << 6) + ((q0 & 16) << 1) + (q0 & 15);
            const uint32_t c = (im << 6) + (gl << 3);
            if (to_image) {
                TT_SFPLOAD(LR0, FMT, AM_0, r);
                TT_SFPLOAD(LR1, FMT, AM_0, r + 2);
                TT_SFPLOAD(LR2, FMT, AM_0, r + 16);
                TT_SFPLOAD(LR3, FMT, AM_0, r + 18);
                TTI_SFPTRANSP(0, 0, 0, 0);
                TT_SFPSTORE(LR0, FMT, AM_0, c);
                TT_SFPSTORE(LR1, FMT, AM_0, c + 2);
                TT_SFPSTORE(LR2, FMT, AM_0, c + 4);
                TT_SFPSTORE(LR3, FMT, AM_0, c + 6);
            } else {
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
    }
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace sppg
#endif

constexpr uint32_t cb_in = 0, cb_a = 1, cb_b = 2, cb_out = 16;
constexpr uint32_t BATCH = 8;

// Batches of up to 8 tiles of cb `src` (front index t0..) -> DST 0..7 -> transpose -> DST 8..15 -> cb `dst`.
inline void transpose_cb(uint32_t src, uint32_t dst, bool to_image) {
    (void)to_image;
    for (uint32_t t0 = 0; t0 < TN; t0 += BATCH) {
        const uint32_t nb = TN - t0 < BATCH ? TN - t0 : BATCH;
        copy_tile_init(src);
        tile_regs_acquire();
        for (uint32_t i = 0; i < nb; ++i) {
            copy_tile(src, t0 + i, i);
        }
        MATH((sppg::transpose_tiles(0, BATCH, nb, to_image)));
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < nb; ++i) {
            // one tile per reservation: cb_out (8 pages) receives 3 x TN tiles, a multi-tile reservation could
            // straddle its wrap point
            cb_reserve_back(dst, 1);
            pack_tile(BATCH + i, dst);
            cb_push_back(dst, 1);
        }
        tile_regs_release();
    }
}

// One max5 stage: TN image tiles of `src` -> DST 0..TN-1 -> pool in place -> `dst` (src popped).
// SPP_MODE 1 (test): no pooling, all three outputs == input (validates transposes and the CB round trips).
inline void stage(uint32_t src, uint32_t dst) {
    cb_wait_front(src, TN);
    copy_tile_init(src);
    tile_regs_acquire();
    for (uint32_t t = 0; t < TN; ++t) {
        copy_tile(src, t, t);
    }
#if SPP_MODE != 1
    MATH((sppg::pool_stage()));
#endif
    tile_regs_commit();
    cb_pop_front(src, TN);
    cb_reserve_back(dst, TN);
    tile_regs_wait();
    for (uint32_t t = 0; t < TN; ++t) {
        pack_tile(t, dst);
    }
    tile_regs_release();
    cb_push_back(dst, TN);
}

void kernel_main() {
    compute_kernel_hw_startup(cb_in, cb_out);
    cb_wait_front(cb_in, TN);
    transpose_cb(cb_in, cb_a, true);
    cb_pop_front(cb_in, TN);

    uint32_t src = cb_a, dst = cb_b;
    for (uint32_t s = 0; s < 3; ++s) {
        stage(src, dst);
        cb_wait_front(dst, TN);
        transpose_cb(dst, cb_out, false);  // max5 / max9 / max13 -> writer
        const uint32_t tmp = src;
        src = dst;
        dst = tmp;
    }
    cb_pop_front(src, TN);
}
