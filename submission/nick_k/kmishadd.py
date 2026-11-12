"""Fused ResBlock tail out = res + mish(y) (ttnn.generic_op), combo_k copy of final_best/kmishadd_v30.py.
Kernels are located relative to this file. Mish math: FB_MA_BF16 [34] (V14 + bf16 round-to-nearest; 13 = V13 lean sfpi) for a bf16 result,
FB_MA_V [30] (V14 raw-TTI replay) for a bfp8 result; then the FPU adds res with DST reused as SrcA.

    out = kmishadd.mish_add(y, res)   # y: residual-branch conv output (pre-activation), res: ResBlock input
    kmishadd.why_not(y, res)          # None if supported, else the reason (fall back to res + mish(y))

out has res's dtype and memory config, like the stock `res + y` it replaces.
"""
import os

import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_R = open(os.path.join(_HERE, "kernels", "reader_push2.cpp")).read()
_SRC_C = open(os.path.join(_HERE, "kernels", "compute_mishadd.cpp")).read()
_DTYPES = (ttnn.bfloat16, ttnn.bfloat8_b)
_BFP8_V = int(os.environ.get("FB_MA_V", "30"))
_BF16_V = int(os.environ.get("FB_MA_BF16", "34"))


def _why1(x):
    if x.layout != ttnn.TILE_LAYOUT:
        return "layout"
    if x.dtype not in _DTYPES:
        return f"dtype {x.dtype}"
    if not x.is_sharded():
        return "interleaved"
    mc = x.memory_config()
    if mc.buffer_type != ttnn.BufferType.L1:
        return "dram"
    sh, sw = mc.shard_spec.shape
    if sh % 32 or sw % 32:
        return f"shard {sh}x{sw}"
    return None


def why_not(y, res):
    r = _why1(y) or _why1(res)
    if r:
        return r
    if y.padded_shape != res.padded_shape:
        return "shape"
    my, mr = y.memory_config(), res.memory_config()
    if my.memory_layout != mr.memory_layout or str(my.shard_spec) != str(mr.shard_spec):
        return "shard spec"
    return None


def supported(y, res):
    return why_not(y, res) is None


def mish_add(y, res, variant=None):
    if variant is None:
        variant = _BF16_V if res.dtype == ttnn.bfloat16 else _BFP8_V
    mc = res.memory_config()
    grid = mc.shard_spec.grid
    sh, sw = mc.shard_spec.shape
    n = (sh // 32) * (sw // 32)
    b = min(8, n)

    out = ttnn.allocate_tensor_on_device(res.shape, res.dtype, ttnn.TILE_LAYOUT, res.device(), mc)
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, y), ttnn.cb_descriptor_from_sharded_tensor(1, res),
           ttnn.cb_descriptor_from_sharded_tensor(16, out)]
    reader = ttnn.KernelDescriptor(
        kernel_source=_SRC_R, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=grid, compile_time_args=[n], runtime_args=[], config=ttnn.ReaderConfigDescriptor())
    ccfg = ttnn.ComputeConfigDescriptor()
    ccfg.fp32_dest_acc_en = False
    ccfg.math_approx_mode = True
    compute = ttnn.KernelDescriptor(
        kernel_source=_SRC_C, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=grid, compile_time_args=[n, b], defines=[("VARIANT", str(variant))], runtime_args=[],
        config=ccfg)
    pd = ttnn.ProgramDescriptor(kernels=[reader, compute], semaphores=[], cbs=cbs)
    return ttnn.generic_op([y, res, out], pd)  # generic_op returns the last io tensor
