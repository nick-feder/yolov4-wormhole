"""Composable patch (combo_k): fuse the ResBlock tail `res + mish(conv3x3(...))` into one custom generic_op
(nick_k/kmishadd.py). Same mechanism as arch_mishadd/patch_mishadd.py:

the residual-branch 3x3 convs (Down1.conv6, DownN.res{k}_conv2; 23 per forward) are tagged when the Down modules are
constructed. A tagged TtConv2d returns its pre-activation output and remembers its Mish post-activation;
ttnn.Tensor.__add__ (the `res + output_tensor` in the downsample __call__s) recognises that tensor and runs
kmishadd.mish_add(y, res) instead of mish + add. If the pair is not supported it falls back to
`res + post_activation(y)` (post_activation = the kmish2 / folder Mish hook). Must be applied before the model is built.

    MISHADD=0   disable (apply() is a no-op)
    FB_MA_V     Mish math for bfp8 results (30 = V14 [default], 12 = V12); bf16 results use FB_MA_BF16 (34 = V14 + RNE [default], 13 = V13)
    NK_LOG=1    print the first 23 fused / fallback calls (descriptors of y and res)
"""
import atexit
import importlib
import os

from . import kmish2, kmishadd

_RES_CONVS = {
    "downsample1": ("Down1", ["conv6"]),
    "downsample2": ("Down2", [f"res{k}_conv2" for k in range(1, 3)]),
    "downsample3": ("Down3", [f"res{k}_conv2" for k in range(1, 9)]),
    "downsample4": ("Down4", [f"res{k}_conv2" for k in range(1, 9)]),
    "downsample5": ("Down5", [f"res{k}_conv2" for k in range(1, 5)]),
}
stats = {"fused": 0, "fallback": 0, "why": {}}
_pending = {}
_applied = False


def apply():
    global _applied
    if _applied or os.environ.get("MISHADD", "1") == "0":
        return stats
    _applied = True
    import ttnn
    import models.tt_cnn.tt.builder as builder

    log = os.environ.get("NK_LOG") == "1"
    for mod, (cls_name, attrs) in _RES_CONVS.items():
        cls = getattr(importlib.import_module(f"models.demos.yolov4.tt.{mod}"), cls_name)
        orig_init = cls.__init__

        def init(self, *a, _orig=orig_init, _attrs=attrs, **k):
            _orig(self, *a, **k)
            for name in _attrs:
                getattr(self, name)._mishadd = True

        cls.__init__ = init

    orig_call = builder.TtConv2d.__call__

    def conv_call(self, x, return_output_dim=False):
        act = self.post_activation
        if not getattr(self, "_mishadd", False) or act is None:
            return orig_call(self, x, return_output_dim)
        self.post_activation = None
        try:
            r = orig_call(self, x, return_output_dim)
        finally:
            self.post_activation = act
        _pending["y"] = r[0] if return_output_dim else r
        _pending["act"] = act
        return r

    builder.TtConv2d.__call__ = conv_call

    orig_add = ttnn.Tensor.__add__

    def tensor_add(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and _pending and args[0] is _pending.get("y"):
            y, act = _pending.pop("y"), _pending.pop("act")
            why = kmishadd.why_not(y, self)
            n = stats["fused"] + stats["fallback"]
            if log and n < 23:
                print(f"[nick_k.mishadd] #{n} y {kmish2.desc(y)} | res {kmish2.desc(self)} -> {why or 'fused'}",
                      flush=True)
            if why is None:
                stats["fused"] += 1
                return kmishadd.mish_add(y, self)
            stats["fallback"] += 1
            stats["why"][why] = stats["why"].get(why, 0) + 1
            return orig_add(self, act(y))
        return orig_add(self, *args, **kwargs)

    ttnn.Tensor.__add__ = tensor_add
    print(f"[nick_k.mishadd] applied (bfp8 math V{kmishadd._BFP8_V}, bf16 V{kmishadd._BF16_V})", flush=True)
    atexit.register(lambda: print(f"[nick_k.mishadd] calls (incl. compile/capture): {stats}", flush=True))
    return stats
