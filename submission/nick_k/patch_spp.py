"""Composable patch: YOLOv4 neck SPP block as ONE custom generic_op (spp.py), all graph sizes.

    patch_spp.apply()        # after patch_cfg (folder submission.py) and after patch_upcat.apply()

Replaces, in TtNeck.__call__, the op sequence between conv3 and conv4
    S2I(conv3) -> 3 x max_pool2d(5/9/13) (-> S2I) -> concat -> to_layout(TILE) [-> add(0, bfp8)] -> S2I(conv3) -> concat
  320 graph (10x10 map, bfp8, with add):           spp.spp(conv3_out)      [1,1,100,2048] bfp8 TILE L1 interleaved
  384/512/640 graphs (12/16/20 map, bf16, no add): spp.spp_g(conv3_out, n) [1,1,n*n,2048] bf16 TILE L1 interleaved
bit-exact (max pooling and concat are exact). Anything else runs the stock ops (STATS["fallback"]).

How: apply() swaps the `ttnn` global of the module that defines the *current* TtNeck.__call__ (stock neck.py, or
patch_upcat's copy) for a proxy that chains to the previous `ttnn` global (e.g. patch_cfg's _Proxy). The proxy defers
the SPP ops with lazy placeholders and runs the fused op at the final concat; anything that does not match the pattern
materializes the placeholders through the real ops (stock behaviour). Apply it AFTER any patch that rebinds
TtNeck.__call__. Kernel sources / spp.py are located relative to this file.

Env:  FB_SPP=1 (0 = proxy installed but passes everything through = stock)
"""
import importlib.util
import math
import os
import sys

import ttnn  # noqa: E402
from models.demos.yolov4.tt import neck as _neck  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name):
    mod_name = f"_combo_neck_{abs(hash(_HERE)) & 0xFFFFFF:x}_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_spp = _load_sibling("spp")


def _map_n(t):
    """10 for the 320 bfp8 path, n for the generic bf16 path, 0 = unsupported."""
    if _spp.supported(t):
        return 10
    rows = t.shape[2]
    n = math.isqrt(rows)
    return n if n * n == rows and _spp.supported_g(t, n) else 0

ENABLED = os.environ.get("FB_SPP", "1") != "0"
STATS = {"custom": 0, "fallback": 0}
_FAILED = set()
_KS = (5, 9, 13)


class _Sent:
    """Placeholder for a deferred stock op result."""

    real = None

    def mat(self):
        raise NotImplementedError

    def memory_config(self):
        return self.mat().memory_config() if self.real is not None else ttnn.L1_MEMORY_CONFIG


class _Lazy(_Sent):
    """sharded_to_interleaved(conv3_out) not yet executed."""

    def __init__(self, px, t, a, kw, n):
        self.px, self.t, self.a, self.kw, self.n = px, t, a, kw, n
        self.shape, self.dtype, self.layout = t.shape, t.dtype, t.layout

    def mat(self):
        if self.real is None:
            self.real = self.px._real.sharded_to_interleaved(self.t, *self.a, **self.kw)
        return self.real


class _Pool(_Sent):
    """max_pool2d(lazy conv3_out, k) not yet executed (an S2I on it is a no-op: the stock output is interleaved)."""

    dtype = ttnn.bfloat16

    def __init__(self, px, src, k, kw):
        self.px, self.src, self.k, self.kw = px, src, k, kw

    def mat(self):
        if self.real is None:
            r = self.px._real
            out = r.max_pool2d(**dict(self.kw, input_tensor=self.src.mat()))
            if out.memory_config().is_sharded():
                out = r.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)
            self.real = out
        return self.real


class _PoolAll(_Sent):
    """concat([pool13, pool9, pool5]) [-> to_layout(TILE) [-> add(0, bfp8)]] not yet executed."""

    def __init__(self, px, pools, ckw):
        self.px, self.pools, self.ckw = px, pools, ckw
        self.steps = []  # ("tile", kw) / ("add", kw)
        self.src = pools[0].src

    @property
    def dtype(self):
        return ttnn.bfloat8_b if any(s == "add" for s, _ in self.steps) else ttnn.bfloat16

    def mat(self):
        if self.real is None:
            r = self.px._real
            out = r.concat([p.mat() for p in self.pools], **self.ckw)
            for s, kw in self.steps:
                out = r.to_layout(out, **kw) if s == "tile" else r.add(out, 0.0, **kw)
            self.real = out
        return self.real


def _m(x):
    return x.mat() if isinstance(x, _Sent) else x


class _Proxy:
    """Stand-in for the neck module's `ttnn` global."""

    def __init__(self, real):
        self.__dict__["_real"] = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def _fallback(self, name, a, kw):
        flat = [v for x in list(a) + list(kw.values()) for v in (x if isinstance(x, (list, tuple)) else [x])]
        if any(isinstance(v, _Sent) for v in flat):
            STATS["fallback"] += 1
        a = [[_m(v) for v in x] if isinstance(x, (list, tuple)) else _m(x) for x in a]
        kw = {k: _m(v) for k, v in kw.items()}
        return getattr(self._real, name)(*a, **kw)

    def sharded_to_interleaved(self, t, *a, **kw):
        if isinstance(t, _Pool):
            return t
        if ENABLED and not isinstance(t, _Sent):
            n = _map_n(t)
            if n:
                return _Lazy(self, t, a, kw, n)
        return self._fallback("sharded_to_interleaved", (t,) + a, kw)

    def max_pool2d(self, *a, **kw):
        src = kw.get("input_tensor", a[0] if a else None)
        k = kw.get("kernel_size", [0, 0])
        ok = (isinstance(src, _Lazy) and not a and k[0] == k[1] and k[0] in _KS
              and list(kw.get("stride", [])) == [1, 1] and list(kw.get("padding", [])) == [k[0] // 2] * 2
              and list(kw.get("dilation", [1, 1])) == [1, 1] and kw.get("input_h") == src.n
              and kw.get("input_w") == src.n and kw.get("batch_size") == 1)
        if ok:
            return _Pool(self, src, k[0], kw)
        return self._fallback("max_pool2d", a, kw)

    def concat(self, tensors, *a, **kw):
        dim = kw.get("dim", a[0] if a else None)
        ts = list(tensors)
        if dim == 3 and len(ts) == 3 and all(isinstance(p, _Pool) and p.real is None for p in ts) \
                and [p.k for p in ts] == [13, 9, 5] and len({id(p.src.t) for p in ts}) == 1:
            return _PoolAll(self, ts, dict(kw, **({"dim": a[0]} if a else {})))
        if dim == 3 and len(ts) == 2 and isinstance(ts[0], _PoolAll) and isinstance(ts[1], _Lazy) \
                and ts[0].real is None and ts[1].real is None and ts[0].src.t is ts[1].t:
            steps = [s for s, _ in ts[0].steps]
            t, n = ts[1].t, ts[1].n
            key = (n, str(t.memory_config()))
            try:
                if key not in _FAILED and steps == ["tile", "add"] and n == 10 and _spp.supported(t):
                    out = _spp.spp(t)
                    STATS["custom"] += 1
                    return out
                if key not in _FAILED and steps == ["tile"] and _spp.supported_g(t, n):
                    out = _spp.spp_g(t, n)
                    STATS["custom"] += 1
                    return out
            except Exception as ex:  # e.g. CBs clash with L1 buffers: remember and run stock from now on
                _FAILED.add(key)
                print(f"[patch_spp] custom SPP failed, stock fallback: {str(ex).splitlines()[0][:200]}", flush=True)
        return self._fallback("concat", (ts,) + a, kw)

    def to_layout(self, t, *a, **kw):
        lay = kw.get("layout", a[0] if a else None)
        if isinstance(t, _PoolAll) and t.real is None and not t.steps and lay == ttnn.TILE_LAYOUT:
            t.steps.append(("tile", dict(kw, **({"layout": a[0]} if a else {}))))
            return t
        return self._fallback("to_layout", (t,) + a, kw)

    def add(self, x, y, *a, **kw):
        if isinstance(x, _PoolAll) and x.real is None and [s for s, _ in x.steps] == ["tile"] and not a \
                and isinstance(y, (int, float)) and y == 0 and kw.get("dtype") == ttnn.bfloat8_b:
            x.steps.append(("add", kw))
            return x
        return self._fallback("add", (x, y) + a, kw)

    def deallocate(self, t, *a, **kw):
        if isinstance(t, _Sent):
            if t.real is not None:
                self._real.deallocate(t.real, *a, **kw)
            return None
        return self._real.deallocate(t, *a, **kw)


_APPLIED = []


def apply():
    if _APPLIED:
        return
    g = _neck.TtNeck.__call__.__globals__
    g["ttnn"] = _Proxy(g["ttnn"])
    _APPLIED.append(g)
    print(f"[patch_spp] ttnn proxy installed in {g.get('__name__')}, FB_SPP={int(ENABLED)}", flush=True)
