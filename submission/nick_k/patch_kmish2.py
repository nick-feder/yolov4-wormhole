"""Composable patch (combo_k): route every Mish of the TtConv2d post-activation hook (builder._POST["HARDMISH"])
through the custom SFPU kernel nick_k/kmish2.py (VARIANT 30 = V14 raw-TTI replay Mish on bfp8, 34 = V14 + bf16 RNE rounding on bf16).

    from nick_k import patch_kmish2; patch_kmish2.apply()

    KM2 [30]         kernel variant for bfp8 tensors (30 V14, 12 V12; 0 = do nothing)
    KM2_BF16 [34]    variant for bf16 tensors (SFPSTORE truncates to bf16: 34 = V14 + SFP_STOCH_RND RNE, 13 = V13 sfpi,
                     0 = bf16 stays on the previous hook, counted as fallback "bf16_off")
    KM2_RANGE [0:72] custom kernel only on hook calls [LO, HI) (index modulo 72)
    NK_LOG=1         print every call of the first forward (tensor descriptor + custom / fallback reason)

Unsupported tensors (not TILE / not L1-sharded / not bf16|bfp8) fall back to whatever hook was installed before
apply() (the folder's ttnn.mish fast-mode hook), so patches stack.
"""
import atexit
import os

from . import kmish2

stats = {"custom": 0, "fallback": 0, "seen": 0, "why": {}}
_applied = False


def apply():
    global _applied
    km = int(os.environ.get("KM2", "30"))
    if _applied or not km:
        print(f"[nick_k.kmish2] KM2={km} (not applied)", flush=True)
        return stats
    import ttnn
    import models.tt_cnn.tt.builder as builder

    km_bf16 = int(os.environ.get("KM2_BF16", "34"))
    lo, hi = (int(v) for v in os.environ.get("KM2_RANGE", "0:72").split(":"))
    log = os.environ.get("NK_LOG") == "1"
    prev = builder._POST.get("HARDMISH", lambda x, p: ttnn.mish(x))

    def _mish(x, p):
        layer = stats["seen"] % 72
        stats["seen"] += 1
        why = None if lo <= layer < hi else "range"
        why = why or kmish2.why_not(x) or ("bf16_off" if not km_bf16 and x.dtype == ttnn.bfloat16 else None)
        if log and stats["seen"] <= 72:
            print(f"[nick_k.kmish2] #{stats['seen'] - 1} {kmish2.desc(x)} -> {why or 'custom'}", flush=True)
        if why is None:
            stats["custom"] += 1
            return kmish2.mish(x, km_bf16 if x.dtype == ttnn.bfloat16 else km, inplace=True)
        stats["fallback"] += 1
        stats["why"][why] = stats["why"].get(why, 0) + 1
        return prev(x, p)

    builder._POST["HARDMISH"] = _mish
    _applied = True
    print(f"[nick_k.kmish2] KM2={km} bf16={km_bf16} RANGE={lo}:{hi}", flush=True)
    atexit.register(lambda: print(f"[nick_k.kmish2] calls (incl. compile/capture): {stats}", flush=True))
    return stats
