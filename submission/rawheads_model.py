"""Raw-heads variant of TtYOLOv4.__call__ (Chanyoung, stream "#1 genboxes removal").

The stock model ends with three TtGenBoxes calls + concat/permute/reshape: ~229 tiny card ops
(~1.1 ms kernels + ~0.5 ms dispatch gaps at 320x320). Here the model stops after the three head
convs and only "packs" the heads so the host can read them cheaply; the box decode moves to the
host (rawheads_decode.py).

Modes (install(mode)):
    orig   stock __call__ (genboxes on card)                -> (boxes, confs)
    raw    head tensors as they leave the convs              -> [h0, h1, h2]  (TILE, sharded L1, h0 bfp8)
    rm     each head untilized to ROW_MAJOR bf16, L1 interl. -> [h0, h1, h2]  [1,1,HW,256]
    rmcat  rm + one concat                                   -> [1,1,2100,256] ROW_MAJOR bf16
Nothing in ~/tt-metal is edited; we only rebind TtYOLOv4.__call__ in this process.
"""
import os

import ttnn
from models.demos.yolov4.tt.yolov4 import TtYOLOv4

_ORIG_CALL = TtYOLOv4.__call__
# RH_MEM=dram: pack the heads in DRAM instead of L1 (larger inputs, e.g. 512x512: 5376 x 256 bf16 = 2.75 MB)
_MEM = ttnn.DRAM_MEMORY_CONFIG if os.environ.get("RH_MEM") == "dram" else ttnn.L1_MEMORY_CONFIG
MODES = ("orig", "raw", "rm", "rmcat")


def _backbone_heads(self, x):
    # identical to the first half of TtYOLOv4.__call__ (pinned 5a3ee49ca03)
    N, C, H, W = x.shape
    min_channels = 16
    if C < min_channels:
        nchw = ttnn.pad(x, ((0, 0), (0, min_channels - C), (0, 0), (0, 0)), value=0.0)
    else:
        nchw = x
    nhwc = ttnn.permute(nchw, (0, 2, 3, 1))
    ttnn.deallocate(nchw)
    ttnn.deallocate(x)
    nhwc = ttnn.reallocate(nhwc)
    x = ttnn.reshape(nhwc, [1, 1, nhwc.shape[0] * nhwc.shape[1] * nhwc.shape[2], nhwc.shape[-1]])
    d1 = self.down1(x)
    d2 = self.down2(d1)
    ttnn.deallocate(d1)
    d3 = self.down3(d2)
    ttnn.deallocate(d2)
    d4 = self.down4(d3)
    d5 = self.down5(d4)
    x20, x13, x6 = self.neck([d5, d4, d3])
    return self.head([x20, x13, x6])


def _to_rm(h):
    if h.dtype != ttnn.bfloat16:
        h = ttnn.typecast(h, ttnn.bfloat16)
    if h.memory_config().is_sharded():
        h = ttnn.sharded_to_interleaved(h, _MEM)
    return ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT, memory_config=_MEM)


def _call_raw(self, x):
    return list(_backbone_heads(self, x))


def _call_rm(self, x):
    return [_to_rm(h) for h in _backbone_heads(self, x)]


def _call_rmcat(self, x):
    hs = [_to_rm(h) for h in _backbone_heads(self, x)]
    return ttnn.concat(hs, dim=2, memory_config=_MEM)


_CALLS = {"orig": _ORIG_CALL, "raw": _call_raw, "rm": _call_rm, "rmcat": _call_rmcat}


def install(mode):
    if mode not in _CALLS:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    TtYOLOv4.__call__ = _CALLS[mode]
