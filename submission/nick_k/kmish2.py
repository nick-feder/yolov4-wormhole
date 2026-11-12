"""Custom Mish op (ttnn.generic_op), combo_k copy of arch_kmish2/kmish2.py. Kernels are located relative to this file.

    y = kmish2.mish(x, variant=30, inplace=True)   # x: TILE layout, sharded in L1 (height / width / block)
    kmish2.why_not(x)                               # None if supported, else the reason (fall back to ttnn.mish)

Every core runs on its own L1 shard (CBs alias the shard, zero-copy), so any shard grid / orientation / uneven last
shard works: the padding rows of a partial last shard are computed too, but they lie inside that core's own shard
allocation and are never read as tensor data. DST batches: min(8, tiles per core), the last batch may be shorter.
"""
import os

import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_R = open(os.path.join(_HERE, "kernels", "reader_push.cpp")).read()
_SRC_C = open(os.path.join(_HERE, "kernels", "compute_mish2.cpp")).read()
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088}


def why_not(x):
    if x.layout != ttnn.TILE_LAYOUT:
        return "layout"
    if x.dtype not in _TILE_BYTES:
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


def supported(x):
    return why_not(x) is None


def desc(t):
    """short tensor descriptor for logs: shape dtype layout memory-layout shard grid."""
    try:
        mc = t.memory_config()
        s = f"{tuple(t.padded_shape)} {str(t.dtype).split('.')[-1]} {str(t.layout).split('.')[-1]} " \
            f"{str(mc.memory_layout).split('.')[-1]} {str(mc.buffer_type).split('.')[-1]}"
        if mc.shard_spec is not None:
            s += f" shard={list(mc.shard_spec.shape)} cores={mc.shard_spec.grid.num_cores()}"
        return s
    except Exception as e:  # noqa: BLE001
        return f"?({e})"


def mish(x, variant=30, inplace=False):
    mc = x.memory_config()
    grid = mc.shard_spec.grid
    sh, sw = mc.shard_spec.shape
    n = (sh // 32) * (sw // 32)
    b = min(8, n)

    out = x if inplace else ttnn.allocate_tensor_on_device(x.shape, x.dtype, ttnn.TILE_LAYOUT, x.device(), mc)
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, x), ttnn.cb_descriptor_from_sharded_tensor(16, out)]
    reader = ttnn.KernelDescriptor(
        kernel_source=_SRC_R, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=grid, compile_time_args=[n], runtime_args=[], config=ttnn.ReaderConfigDescriptor())
    ccfg = ttnn.ComputeConfigDescriptor()
    ccfg.fp32_dest_acc_en = False
    ccfg.math_approx_mode = True
    compute = ttnn.KernelDescriptor(
        kernel_source=_SRC_C, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=grid, compile_time_args=[n, b], defines=[("VARIANT", str(variant))],
        runtime_args=[], config=ccfg)
    pd = ttnn.ProgramDescriptor(kernels=[reader, compute], semaphores=[], cbs=cbs)
    io = [x, x] if inplace else [x, out]  # generic_op needs >= 2 io tensors
    return ttnn.generic_op(io, pd)
