// Custom kernel (team Nick): YOLOv4 ResBlock tail  out = res + mish(y)  in one pass.
//   y   = pre-activation output of the residual 3x3 conv (CB 0, aliases its L1 shard)
//   res = ResBlock input (CB 1, aliases its L1 shard, same shard spec as y)
//   out = CB 16 (new tensor with res's dtype and memory config, like the stock add)
// Per block of B tiles: unpack y -> DST, lean Mish on the SFPU (kmish V12/V13 math), then the FPU adds
// the res tile with DST reused as SrcA (add_reuse_dest_tiles<DEST_TO_SRCA>: DST -> SrcA, res -> SrcB),
// then pack.  Replaces kmish + binary_ng add (one op launch and one bfp8 round trip of mish(y) saved).
// final_best copy: + VARIANT 30 = V14 raw-TTI replay Mish from arch_kmish2/compute_mish2.cpp
// VARIANT 12: lean Mish, DST truncation   13: lean Mish + explicit bf16 rounding before the add
//         0: no Mish (out = res + y), overhead/correctness probe of the add path
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/mish.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
#include "ckernel_sfpu_exp.h"
namespace ckernel::sfpu {

// Schraudolph exp via magic-add, magic-constant reciprocal + one tuned Newton step, sign folded into the
// final MAD (same constants and error as kmish V6/V12, see kernels/compute_mish.cpp).
inline void lean_consts2() {
    sfpi::vConstFloatPrgm0 = 47274.98046875f;  // 2^15/ln2
    sfpi::vConstFloatPrgm1 = 12548715.0f;      // 126.9564*2^15 + 2^23
    sfpi::vConstFloatPrgm2 = 0x1.ECp+126f;     // bit pattern 0x7EF60000: reciprocal magic constant
}

sfpi_inline sfpi::vFloat mish_v6(sfpi::vFloat x) {
    sfpi::vFloat xc = sfpi::clamp(x, -80.0f, 8.0f);
    sfpi::vFloat f = xc * sfpi::vConstFloatPrgm0 + sfpi::vConstFloatPrgm1;
    sfpi::vFloat s = sfpi::as<sfpi::vFloat>(sfpi::shft(sfpi::as<sfpi::vUInt>(f), 8)) + sfpi::vConst1;
    sfpi::vFloat d = s * s + sfpi::vConst1;
    sfpi::vFloat m = sfpi::vConstFloatPrgm2;
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(m) - sfpi::as<sfpi::vInt>(d));
    y = y * (d * y - 2.0015878677368164f);
    return x + (x + x) * y;
}

// Two DST rows per iteration as independent dependency chains (fills SFPU latency slots).
template <bool ROUND>
inline void lean_mish_face_x2() {
    for (int i = 0; i < 8; i += 2) {
        sfpi::vFloat a = mish_v6(sfpi::dst_reg[i]);
        sfpi::vFloat b = mish_v6(sfpi::dst_reg[i + 1]);
        if constexpr (ROUND) {
            a = sfpi::convert<sfpi::vFloat16b>(a, sfpi::RoundMode::Nearest);
            b = sfpi::convert<sfpi::vFloat16b>(b, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[i] = a;
        sfpi::dst_reg[i + 1] = b;
    }
}
inline void lean_mish_face_v12() { lean_mish_face_x2<false>(); }
inline void lean_mish_face_v13() { lean_mish_face_x2<true>(); }

// ---------------------------------------------------------------------------------------------------------
// V30 (V14) / V31 (V14-lite): hand-scheduled raw-TTI Mish, two DST rows (chain A: L0-L2 at dst+0,
// chain B: L3-L5 at dst+2) strictly alternating so every dependent instruction is >= 2 slots after its
// producer (MAD/ADD results are not interlocked and need 2 cycles). One row-pair body is recorded into the
// replay buffer (slot 0) and replayed 3 more times per face. Numpy model: ../res_sfpu/sim_v14.py, simchk.py.
//   exp:  t = x*A + B ; n = STOCHRND->UINT16(|t|) (saturates at 65535 = free upper clamp, x >~ 8)
//         u' = bits(n << 14) = 2^(n/512 - 127) = e^x / 2^10.5
//   V30:  s = u'*2^10.5 + 1 ; d = s*s + 1 ; yh = (M + 2^31) - bits(d) ~ -1/d ;
//         e = d*yh + K1 ; g = yh*e + yh (~ -1/d) ; r = (x*1 + x)*g + x     -> 13 instructions/row
//   V31:  works on d/2 with the Newton bias folded in; drops the 2x step -> 12 instructions/row
// Without a lower clamp, t < 0 below x = -80.7 and |t| mirrors: |err| > .01 only for x < -110.5
// (real YOLOv4 Mish inputs reach -54 at most, kdiag9b.log). V32 adds a lower clamp at -80.
#define V14_L_SEED 6  // V30: M'' (sign-folded reciprocal magic), V31: C2
#define V14_L_K1 7    // V30: Newton constant K1, V31: H
#define V14_ONE 10    // fixed constant register 1.0
#define V14_LO_MOD 10  // SFPLOADI mod0: write low 16 bits
#define V14_HI_MOD 8   // SFPLOADI mod0: write high 16 bits

#define V14_LOAD32(reg, v)                                  \
    TTI_SFPLOADI(reg, V14_LO_MOD, ((v) & 0xFFFF));          \
    TTI_SFPLOADI(reg, V14_HI_MOD, (((v) >> 16) & 0xFFFF));
#define V14_SETCONST(creg, v) \
    V14_LOAD32(0, v)          \
    TTI_SFPCONFIG(0, creg, 0);

inline void v14_consts() {
    V14_SETCONST(12, 0x4438AA3Bu)  // A = 512/ln2
    V14_SETCONST(13, 0x4768E666u)  // B = 512*(127 - 10.5 - 0.05)
    V14_SETCONST(14, 0x44B504F3u)  // 2^10.5
    V14_LOAD32(V14_L_SEED, 0xFEF7A000u)  // M'' = 0x7EF7A000 + 2^31
    V14_LOAD32(V14_L_K1, 0x3F802752u)    // K1 = 1.0012
}

inline void v14lite_consts() {
    V14_SETCONST(12, 0x4438AA3Bu)  // A
    V14_SETCONST(13, 0x4768E666u)  // B
    V14_SETCONST(14, 0x447FDF39u)  // C1 = 2^10.5 * sqrt(beta/2), beta = 0.999
    V14_SETCONST(11, 0xFEF6A000u)  // M'' (clobbers sfpi vConstNeg1: restored by v14_restore_neg1)
    V14_LOAD32(V14_L_SEED, 0x3F34EDC6u)  // C2 = sqrt(beta/2)
    V14_LOAD32(V14_L_K1, 0x3EFFBE77u)    // H = beta/2
}

inline void v14_restore_neg1() {
    TTI_SFPLOADI(0, 0 /*FLOATB*/, 0xBF80);  // -1.0
    TTI_SFPCONFIG(0, 11, 0);
}

#define V14_STOCH(r) TTI_SFP_STOCH_RND(0, 0, 0, r, r, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);

// 27 instructions: 26 SFPU + INCRWC (dst += 2 rows of 32 lanes)
#define V14_BODY                                                                          \
    TTI_SFPLOAD(0, 0, ADDR_MOD_3, 0);                                                     \
    TTI_SFPLOAD(3, 0, ADDR_MOD_3, 2);                                                     \
    TTI_SFPMAD(0, 12, 13, 1, 0);                                                          \
    TTI_SFPMAD(3, 12, 13, 4, 0);                                                          \
    V14_STOCH(1)                                                                          \
    V14_STOCH(4)                                                                          \
    TTI_SFPSHFT(14, 1, 1, 1);                                                             \
    TTI_SFPSHFT(14, 4, 4, 1);                                                             \
    TTI_SFPMAD(1, 14, V14_ONE, 1, 0);           /* s = u'*2^K + 1 */                      \
    TTI_SFPMAD(4, 14, V14_ONE, 4, 0);                                                     \
    TTI_SFPMAD(1, 1, V14_ONE, 1, 0);            /* d = s*s + 1 */                         \
    TTI_SFPMAD(4, 4, V14_ONE, 4, 0);                                                      \
    TTI_SFPMOV(0, 1, 2, 0);                                                               \
    TTI_SFPMOV(0, 4, 5, 0);                                                               \
    TTI_SFPIADD(0, V14_L_SEED, 2, 6);           /* yh = M'' - bits(d) */                  \
    TTI_SFPIADD(0, V14_L_SEED, 5, 6);                                                     \
    TTI_SFPMAD(1, 2, V14_L_K1, 1, 0);           /* e = d*yh + K1 */                       \
    TTI_SFPMAD(4, 5, V14_L_K1, 4, 0);                                                     \
    TTI_SFPMAD(2, 1, 2, 1, 0);                  /* g = yh*e + yh */                       \
    TTI_SFPMAD(5, 4, 5, 4, 0);                                                            \
    TTI_SFPMAD(0, V14_ONE, 0, 2, 0);            /* 2x */                                  \
    TTI_SFPMAD(3, V14_ONE, 3, 5, 0);                                                      \
    TTI_SFPMAD(2, 1, 0, 0, 0);                  /* r = 2x*g + x */                        \
    TTI_SFPMAD(5, 4, 3, 3, 0);                                                            \
    TTI_SFPSTORE(0, 0, ADDR_MOD_3, 0);                                                    \
    TTI_SFPSTORE(3, 0, ADDR_MOD_3, 2);                                                    \
    TTI_INCRWC(0, 4, 0, 0);

// 25 instructions: 24 SFPU + INCRWC
#define V14LITE_BODY                                                                      \
    TTI_SFPLOAD(0, 0, ADDR_MOD_3, 0);                                                     \
    TTI_SFPLOAD(3, 0, ADDR_MOD_3, 2);                                                     \
    TTI_SFPMAD(0, 12, 13, 1, 0);                                                          \
    TTI_SFPMAD(3, 12, 13, 4, 0);                                                          \
    V14_STOCH(1)                                                                          \
    V14_STOCH(4)                                                                          \
    TTI_SFPSHFT(14, 1, 1, 1);                                                             \
    TTI_SFPSHFT(14, 4, 4, 1);                                                             \
    TTI_SFPMAD(1, 14, V14_L_SEED, 1, 0);        /* s_h = u'*C1 + C2 */                    \
    TTI_SFPMAD(4, 14, V14_L_SEED, 4, 0);                                                  \
    TTI_SFPMAD(1, 1, V14_L_K1, 1, 0);           /* d_h = s_h^2 + H */                     \
    TTI_SFPMAD(4, 4, V14_L_K1, 4, 0);                                                     \
    TTI_SFPMOV(0, 1, 2, 0);                                                               \
    TTI_SFPMOV(0, 4, 5, 0);                                                               \
    TTI_SFPIADD(0, 11, 2, 6);                   /* yn = M'' - bits(d_h) ~ -2/d */         \
    TTI_SFPIADD(0, 11, 5, 6);                                                             \
    TTI_SFPMAD(1, 2, V14_ONE, 1, 0);            /* e = d_h*yn + 1 */                      \
    TTI_SFPMAD(4, 5, V14_ONE, 4, 0);                                                      \
    TTI_SFPMAD(2, 1, 2, 1, 0);                  /* f = yn*e + yn */                       \
    TTI_SFPMAD(5, 4, 5, 4, 0);                                                            \
    TTI_SFPMAD(0, 1, 0, 0, 0);                  /* r = x*f + x */                         \
    TTI_SFPMAD(3, 4, 3, 3, 0);                                                            \
    TTI_SFPSTORE(0, 0, ADDR_MOD_3, 0);                                                    \
    TTI_SFPSTORE(3, 0, ADDR_MOD_3, 2);                                                    \
    TTI_INCRWC(0, 4, 0, 0);

// One face = 8 SFPU rows = 4 row-pairs: record + execute the first pair, replay the other 3.
// The face-advance SETRWC (relative to the carriage-return counter) discards the INCRWCs.
inline void v14_face() {
    TTI_REPLAY(0, 27, 1, 1);
    V14_BODY
    TTI_REPLAY(0, 27, 0, 0);
    TTI_REPLAY(0, 27, 0, 0);
    TTI_REPLAY(0, 27, 0, 0);
}

// combo_k V34 = V14 + round-to-nearest-even to bf16 before the store (SFPSTORE to 16-bit DST truncates).
#define V34_RND16(r) TTI_SFP_STOCH_RND(0 /*RND_EVEN*/, 0, 0, r, r, 1 /*FP32_TO_FP16B*/);
#define V34_BODY                                                                          \
    TTI_SFPLOAD(0, 0, ADDR_MOD_3, 0);                                                     \
    TTI_SFPLOAD(3, 0, ADDR_MOD_3, 2);                                                     \
    TTI_SFPMAD(0, 12, 13, 1, 0);                                                          \
    TTI_SFPMAD(3, 12, 13, 4, 0);                                                          \
    V14_STOCH(1)                                                                          \
    V14_STOCH(4)                                                                          \
    TTI_SFPSHFT(14, 1, 1, 1);                                                             \
    TTI_SFPSHFT(14, 4, 4, 1);                                                             \
    TTI_SFPMAD(1, 14, V14_ONE, 1, 0);                                                     \
    TTI_SFPMAD(4, 14, V14_ONE, 4, 0);                                                     \
    TTI_SFPMAD(1, 1, V14_ONE, 1, 0);                                                      \
    TTI_SFPMAD(4, 4, V14_ONE, 4, 0);                                                      \
    TTI_SFPMOV(0, 1, 2, 0);                                                               \
    TTI_SFPMOV(0, 4, 5, 0);                                                               \
    TTI_SFPIADD(0, V14_L_SEED, 2, 6);                                                     \
    TTI_SFPIADD(0, V14_L_SEED, 5, 6);                                                     \
    TTI_SFPMAD(1, 2, V14_L_K1, 1, 0);                                                     \
    TTI_SFPMAD(4, 5, V14_L_K1, 4, 0);                                                     \
    TTI_SFPMAD(2, 1, 2, 1, 0);                                                            \
    TTI_SFPMAD(5, 4, 5, 4, 0);                                                            \
    TTI_SFPMAD(0, V14_ONE, 0, 2, 0);                                                      \
    TTI_SFPMAD(3, V14_ONE, 3, 5, 0);                                                      \
    TTI_SFPMAD(2, 1, 0, 0, 0);                                                            \
    TTI_SFPMAD(5, 4, 3, 3, 0);                                                            \
    V34_RND16(0)                                                                          \
    V34_RND16(3)                                                                          \
    TTI_SFPSTORE(0, 0, ADDR_MOD_3, 0);                                                    \
    TTI_SFPSTORE(3, 0, ADDR_MOD_3, 2);                                                    \
    TTI_INCRWC(0, 4, 0, 0);

inline void v34_face() {
    TTI_REPLAY(0, 29, 1, 1);
    V34_BODY
    TTI_REPLAY(0, 29, 0, 0);
    TTI_REPLAY(0, 29, 0, 0);
    TTI_REPLAY(0, 29, 0, 0);
}


}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    constexpr uint32_t n_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);  // tiles per DST batch (<= 8); last batch may be shorter
    constexpr uint32_t cb_y = 0, cb_res = 1, cb_out = 16;

    // SrcA <- y (copy_tile), SrcB <- res (dest-reuse add), packer -> out format.
    compute_kernel_hw_startup(cb_y, cb_res, cb_out);
#if VARIANT != 0
    mish_tile_init<true>();  // SFPU config; the lean constants below replace the library's
#if VARIANT == 30 || VARIANT == 34
    MATH((ckernel::sfpu::v14_consts()));  // V14 constants (L12-14 via SFPCONFIG, L6/L7); FPU add leaves them alone
#else
    MATH((ckernel::sfpu::lean_consts2()));
#endif
#endif

    cb_wait_front(cb_y, n_tiles);
    cb_wait_front(cb_res, n_tiles);
    for (uint32_t t = 0; t < n_tiles; t += B) {
        const uint32_t nb = (n_tiles - t < B) ? n_tiles - t : B;
        tile_regs_acquire();
        copy_tile_init(cb_y);
        for (uint32_t i = 0; i < nb; ++i) {
            copy_tile(cb_y, t + i, i);
        }
#if VARIANT != 0
        for (uint32_t i = 0; i < nb; ++i) {
#if VARIANT == 13
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v13, i,
                                                  VectorMode::RC));
#elif VARIANT == 30
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, v14_face, i, VectorMode::RC));
#elif VARIANT == 34
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, v34_face, i, VectorMode::RC));
#else
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v12, i,
                                                  VectorMode::RC));
#endif
        }
#ifdef SFPU_STALL
        // FPU must not move DST -> SrcA before the SFPU has written its last row.
        MATH(TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::WAIT_SFPU));
#endif
#endif
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_res);
        for (uint32_t i = 0; i < nb; ++i) {
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_res, t + i, i);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, nb);
        tile_regs_wait();
        for (uint32_t i = 0; i < nb; ++i) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, nb);
    }
}
