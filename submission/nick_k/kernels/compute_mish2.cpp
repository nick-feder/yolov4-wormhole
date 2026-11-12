// Custom kernel (team Nick): Mish on the SFPU with a leaner instruction stream than the library op.
//
// Library fast Mish per 32-lane row: v_if saturation, exp_21f (float->int, exexp/exman, int->float,
// degree-2 polynomial, setexp, bf16 rounding), reciprocal (quadratic seed + Newton step + sign fixup),
// final bf16 rounding.  ~50 SFPU instructions.
// Lean Mish: clamp instead of v_if; Schraudolph exp (the IEEE-754 bit trick: int(x/ln2+127)*2^23
// reinterpreted as float, i.e. the first 4 instructions of exp_21f without the polynomial);
// no intermediate bf16 rounding (the packer rounds to bfp8 anyway).
//   mish(x) = x * (1 - 2 / ((1+u)^2 + 1)),  u = exp(min(max(x, -80), 8))
// VARIANT 0: copy only (overhead floor)   1: library mish_tile<true>
//         2: lean, library reciprocal       3: lean, magic-constant reciprocal + 2 Newton steps
// arch_kmish2 copy: adds VARIANT 30 (V14 raw-TTI replay, 13 instr/row), 31 (V14-lite, 12 instr/row),
//         33 (V12 with the Newton constant in a constant register)
// combo_k copy: adds VARIANT 34 (V14 + bf16 round-to-nearest-even, for bf16 tensors) and a shorter last DST batch
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/mish.h"
#include "api/compute/compute_kernel_api.h"

// 20 = V8 + restore library constants at exit, 21 = library math + clobber constants at exit (diagnostics)
#if VARIANT == 20
#define RESTORE_CONSTS 1
#undef VARIANT
#define VARIANT 8
#elif VARIANT == 21
#define CLOBBER_CONSTS 1
#undef VARIANT
#define VARIANT 1
#endif
#ifdef TRISC_MATH
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat lean_exp(sfpi::vFloat x) {
    sfpi::vFloat xc = sfpi::clamp(x, -80.0f, 8.0f);
    sfpi::vFloat t = xc * 1.4426950409f + 126.9564f;  // 127 - 0.0436: Schraudolph bias, halves max error
    return sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(t));
}

inline void lean_mish_face_v2() {
    for (int i = 0; i < 8; i++) {
        sfpi::vFloat x = sfpi::dst_reg[i];
        sfpi::vFloat s = lean_exp(x) + 1.0f;
        sfpi::vFloat r = sfpu_reciprocal_iter<0>(s * s + 1.0f);
        sfpi::dst_reg[i] = x - (x + x) * r;
    }
}

inline void lean_mish_face_v3() {
    for (int i = 0; i < 8; i++) {
        sfpi::vFloat x = sfpi::dst_reg[i];
        sfpi::vFloat s = lean_exp(x) + 1.0f;
        sfpi::vFloat d = s * s + 1.0f;  // d >= 1, positive: no sign handling needed
        sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x7EF311C3) - sfpi::as<sfpi::vInt>(d));
        y = y * (2.0f - d * y);
        y = y * (2.0f - d * y);
        sfpi::dst_reg[i] = x - (x + x) * y;
    }
}

// V4/V5: all three constants live in the programmable constant registers (no SFPLOADI in the loop),
// reciprocal = magic-constant seed + one Newton step with a tuned constant (max rel err 0.34% for d>=1),
// sign folded into the final MAD:  y = -(1/d),  mish = x + 2x*y.
inline void lean_consts() {
    sfpi::vConstFloatPrgm0 = 1.4426950409f;  // 1/ln2
    sfpi::vConstFloatPrgm1 = 126.9564f;      // 127 - Schraudolph bias
    sfpi::vConstFloatPrgm2 = 0x1.E98p+126f;  // bit pattern 0x7EF4C000: reciprocal magic constant
}

template <bool ROUND>
inline void lean_mish_face_v4() {
    for (int i = 0; i < 8; i++) {
        sfpi::vFloat x = sfpi::dst_reg[i];
        sfpi::vFloat xc = sfpi::clamp(x, -80.0f, 8.0f);
        sfpi::vFloat t = xc * sfpi::vConstFloatPrgm0 + sfpi::vConstFloatPrgm1;
        sfpi::vFloat s = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(t)) + sfpi::vConst1;
        sfpi::vFloat d = s * s + sfpi::vConst1;
        sfpi::vFloat m = sfpi::vConstFloatPrgm2;
        sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(m) - sfpi::as<sfpi::vInt>(d));
        y = y * (d * y - 2.001f);
        sfpi::vFloat r = x + (x + x) * y;
        if constexpr (ROUND) {
            r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[i] = r;
    }
}
inline void lean_mish_face_v4n() { lean_mish_face_v4<false>(); }
inline void lean_mish_face_v4r() { lean_mish_face_v4<true>(); }

// V6/V7: Schraudolph exp via a magic-add instead of exexp/exman/shft:
//   f = x*(2^15/ln2) + (126.9564*2^15 + 2^23)  ->  mantissa(f) = round(t*2^15), exponent(f) = 150
//   as_int(f) << 8 = round(t*2^15)*2^8 = Schraudolph bits of 2^t  (exponent 150 shifts out; 150 is even)
// Reciprocal constant `a` is chosen so that 1/d is exact at d = 2 (x -> -inf), so the error does not
// grow with |x| for negative inputs.
inline void lean_consts2() {
    sfpi::vConstFloatPrgm0 = 47274.98046875f;     // 2^15/ln2
    sfpi::vConstFloatPrgm1 = 12548715.0f;         // 126.9564*2^15 + 2^23
    sfpi::vConstFloatPrgm2 = 0x1.ECp+126f;        // bit pattern 0x7EF60000: reciprocal magic constant
}

template <bool ROUND>
inline void lean_mish_face_v6() {
    for (int i = 0; i < 8; i++) {
        sfpi::vFloat x = sfpi::dst_reg[i];
        sfpi::vFloat xc = sfpi::clamp(x, -80.0f, 8.0f);
        sfpi::vFloat f = xc * sfpi::vConstFloatPrgm0 + sfpi::vConstFloatPrgm1;
        sfpi::vFloat u = sfpi::as<sfpi::vFloat>(sfpi::shft(sfpi::as<sfpi::vUInt>(f), 8));
        sfpi::vFloat s = u + sfpi::vConst1;
        sfpi::vFloat d = s * s + sfpi::vConst1;
        sfpi::vFloat m = sfpi::vConstFloatPrgm2;
        sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(m) - sfpi::as<sfpi::vInt>(d));
        y = y * (d * y - 2.0015878677368164f);
        sfpi::vFloat r = x + (x + x) * y;
        if constexpr (ROUND) {
            r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[i] = r;
    }
}
inline void lean_mish_face_v6n() { lean_mish_face_v6<false>(); }
inline void lean_mish_face_v6r() { lean_mish_face_v6<true>(); }

// V8/V9: accurate variant (max |err| 4.5e-4 vs library fast Mish 8e-4).
//   exp: Cody-Waite.  f = x/ln2 + 1.5*2^23 puts n = round(x/ln2) in the mantissa; r = x/ln2 - n in [-.5,.5];
//        2^r by a degree-2 polynomial (rel err 1.7e-3); scale by 2^n by integer-adding (bits(f) << 23).
//   1/d: magic seed + 2 Newton steps, sign-folded so every step is a plain MAD.
inline void lean_consts3() {
    sfpi::vConstFloatPrgm0 = 1.4426950409f;  // 1/ln2
    sfpi::vConstFloatPrgm1 = 12582912.0f;    // 1.5*2^23 (round-to-integer magic)
    sfpi::vConstFloatPrgm2 = 0x1.E7p+126f;   // bit pattern 0x7EF38000: reciprocal magic constant
}

template <bool ROUND>
inline void lean_mish_face_v8() {
    for (int i = 0; i < 8; i++) {
        sfpi::vFloat x = sfpi::dst_reg[i];
        sfpi::vFloat xc = sfpi::clamp(x, -80.0f, 8.0f);
        sfpi::vFloat f = xc * sfpi::vConstFloatPrgm0 + sfpi::vConstFloatPrgm1;
        sfpi::vFloat nneg = sfpi::vConstFloatPrgm1 - f;
        sfpi::vFloat r = xc * sfpi::vConstFloatPrgm0 + nneg;
        sfpi::vFloat p = (r * 0.23841564f + 0.70342398f) * r + 1.00044222f;
        sfpi::vInt e = sfpi::as<sfpi::vInt>(sfpi::shft(sfpi::as<sfpi::vUInt>(f), 23));
        sfpi::vFloat u = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(p) + e);
        sfpi::vFloat s = u + sfpi::vConst1;
        sfpi::vFloat d = s * s + sfpi::vConst1;
        sfpi::vFloat m = sfpi::vConstFloatPrgm2;
        sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(m) - sfpi::as<sfpi::vInt>(d));
        y = y * (d * y - 2.0f);  // -1/d after step 1
        y = y * (d * y + 2.0f);  // -1/d after step 2
        sfpi::vFloat res = x + (x + x) * y;
        if constexpr (ROUND) {
            res = sfpi::convert<sfpi::vFloat16b>(res, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[i] = res;
    }
}
inline void lean_mish_face_v8n() { lean_mish_face_v8<false>(); }
inline void lean_mish_face_v8r() { lean_mish_face_v8<true>(); }

// V10-V13: same math as V8 / V6 but two DST rows per iteration as independent dependency chains,
// so the compiler can fill SFPU pipeline-latency slots of one chain with the other chain's instructions.
sfpi_inline sfpi::vFloat mish_v8(sfpi::vFloat x) {
    sfpi::vFloat xc = sfpi::clamp(x, -80.0f, 8.0f);
    sfpi::vFloat f = xc * sfpi::vConstFloatPrgm0 + sfpi::vConstFloatPrgm1;
    sfpi::vFloat nneg = sfpi::vConstFloatPrgm1 - f;
    sfpi::vFloat r = xc * sfpi::vConstFloatPrgm0 + nneg;
    sfpi::vFloat p = (r * 0.23841564f + 0.70342398f) * r + 1.00044222f;
    sfpi::vInt e = sfpi::as<sfpi::vInt>(sfpi::shft(sfpi::as<sfpi::vUInt>(f), 23));
    sfpi::vFloat s = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(p) + e) + sfpi::vConst1;
    sfpi::vFloat d = s * s + sfpi::vConst1;
    sfpi::vFloat m = sfpi::vConstFloatPrgm2;
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(m) - sfpi::as<sfpi::vInt>(d));
    y = y * (d * y - 2.0f);
    y = y * (d * y + 2.0f);
    return x + (x + x) * y;
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

template <bool ROUND, bool V8>
inline void lean_mish_face_x2() {
    for (int i = 0; i < 8; i += 2) {
        sfpi::vFloat a = sfpi::dst_reg[i];
        sfpi::vFloat b = sfpi::dst_reg[i + 1];
        if constexpr (V8) {
            a = mish_v8(a);
            b = mish_v8(b);
        } else {
            a = mish_v6(a);
            b = mish_v6(b);
        }
        if constexpr (ROUND) {
            a = sfpi::convert<sfpi::vFloat16b>(a, sfpi::RoundMode::Nearest);
            b = sfpi::convert<sfpi::vFloat16b>(b, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[i] = a;
        sfpi::dst_reg[i + 1] = b;
    }
}
inline void lean_mish_face_v10() { lean_mish_face_x2<false, true>(); }
inline void lean_mish_face_v11() { lean_mish_face_x2<true, true>(); }
inline void lean_mish_face_v12() { lean_mish_face_x2<false, false>(); }
inline void lean_mish_face_v13() { lean_mish_face_x2<true, false>(); }

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

// combo_k V34 = V14 + round-to-nearest-even to bf16 before the store (for bf16 DST/outputs: SFPSTORE would
// truncate). 29 instructions: the two SFP_STOCH_RND(FP32_TO_FP16B, RND_EVEN) sit between the final MADs
// and the stores, each >= 2 slots after its producer and before its consumer.
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

inline void v14lite_face() {
    TTI_REPLAY(0, 25, 1, 1);
    V14LITE_BODY
    TTI_REPLAY(0, 25, 0, 0);
    TTI_REPLAY(0, 25, 0, 0);
    TTI_REPLAY(0, 25, 0, 0);
}

// V33: V12 sfpi cleanup -- same math as V12, but the Newton constant -2.0016 lives in L11 (sfpi
// vConstNeg1, overwritten for the duration of the kernel) instead of being re-materialised every row.
inline void v12c_consts() {
    lean_consts2();
    V14_SETCONST(11, 0xC0001A04u)  // -2.0015878677368164f
}

sfpi_inline sfpi::vFloat mish_v6c(sfpi::vFloat x) {
    sfpi::vFloat xc = sfpi::clamp(x, -80.0f, 8.0f);
    sfpi::vFloat f = xc * sfpi::vConstFloatPrgm0 + sfpi::vConstFloatPrgm1;
    sfpi::vFloat s = sfpi::as<sfpi::vFloat>(sfpi::shft(sfpi::as<sfpi::vUInt>(f), 8)) + sfpi::vConst1;
    sfpi::vFloat d = s * s + sfpi::vConst1;
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(sfpi::vConstFloatPrgm2) - sfpi::as<sfpi::vInt>(d));
    y = y * (d * y + sfpi::vConstNeg1);
    return x + (x + x) * y;
}

inline void lean_mish_face_v33() {
    for (int i = 0; i < 8; i += 2) {
        sfpi::vFloat a = sfpi::dst_reg[i];
        sfpi::vFloat b = sfpi::dst_reg[i + 1];
        sfpi::dst_reg[i] = mish_v6c(a);
        sfpi::dst_reg[i + 1] = mish_v6c(b);
    }
}

}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    constexpr uint32_t n_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);  // tiles per DST batch; the last batch may be shorter
    constexpr uint32_t cb_in = 0, cb_out = 16;

    init_sfpu(cb_in, cb_out);
#if VARIANT >= 1
    mish_tile_init<true>();  // SFPU config + reciprocal seed constants (V4/V5 overwrite the constants)
#endif
#if VARIANT == 4 || VARIANT == 5
    MATH((ckernel::sfpu::lean_consts()));
#elif VARIANT == 6 || VARIANT == 7 || VARIANT == 12 || VARIANT == 13
    MATH((ckernel::sfpu::lean_consts2()));
#elif VARIANT == 30 || VARIANT == 34
    MATH((ckernel::sfpu::v14_consts()));
#elif VARIANT == 31
    MATH((ckernel::sfpu::v14lite_consts()));
#elif VARIANT == 33
    MATH((ckernel::sfpu::v12c_consts()));
#elif VARIANT >= 8
    MATH((ckernel::sfpu::lean_consts3()));
#endif

    cb_wait_front(cb_in, n_tiles);
    for (uint32_t t = 0; t < n_tiles; t += B) {
        const uint32_t nb = (n_tiles - t < B) ? n_tiles - t : B;
        tile_regs_acquire();
        for (uint32_t i = 0; i < nb; ++i) {
            copy_tile(cb_in, t + i, i);
        }
        for (uint32_t i = 0; i < nb; ++i) {
#if VARIANT == 1
            mish_tile<true>(i);
#elif VARIANT == 2
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v2, i,
                                                  VectorMode::RC));
#elif VARIANT == 3
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v3, i,
                                                  VectorMode::RC));
#elif VARIANT == 4
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v4n, i,
                                                  VectorMode::RC));
#elif VARIANT == 5
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v4r, i,
                                                  VectorMode::RC));
#elif VARIANT == 6
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v6n, i,
                                                  VectorMode::RC));
#elif VARIANT == 7
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v6r, i,
                                                  VectorMode::RC));
#elif VARIANT == 8
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v8n, i,
                                                  VectorMode::RC));
#elif VARIANT == 9
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v8r, i,
                                                  VectorMode::RC));
#elif VARIANT == 10
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v10, i,
                                                  VectorMode::RC));
#elif VARIANT == 11
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v11, i,
                                                  VectorMode::RC));
#elif VARIANT == 12
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v12, i,
                                                  VectorMode::RC));
#elif VARIANT == 13
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v13, i,
                                                  VectorMode::RC));
#elif VARIANT == 30
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, v14_face, i, VectorMode::RC));
#elif VARIANT == 34
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, v34_face, i, VectorMode::RC));
#elif VARIANT == 31
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, v14lite_face, i, VectorMode::RC));
#elif VARIANT == 33
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, lean_mish_face_v33, i,
                                                  VectorMode::RC));
#endif
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
#ifdef RESTORE_CONSTS
    mish_tile_init<true>();  // leave the library's reciprocal constants in vConstFloatPrgm0..2
#endif
#ifdef CLOBBER_CONSTS
    MATH((ckernel::sfpu::lean_consts3()));
#endif
#if VARIANT == 31 || VARIANT == 33
    MATH((ckernel::sfpu::v14_restore_neg1()));  // L11 is sfpi vConstNeg1 for any later SFPU code
#endif
}
