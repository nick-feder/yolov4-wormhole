"""Fused YOLOv4 SPP block (ttnn.generic_op): maxpool 5/9/13 (stride 1, same padding) + channel concat.

    out = spp(x)            320 graph (verbatim arch_spp version)
      x    [1,1,100,512] bfp8_b TILE, L1-sharded (neck conv3 output, 10x10 map, rows = positions)
      out  [1,1,100,2048] bfp8_b TILE, L1 interleaved
           == concat([maxpool13(x), maxpool9(x), maxpool5(x), x], dim=3), the tensor the stock neck feeds conv4.
    out = spp_g(x, n)       384/512/640 graphs (n = 12/16/20), bf16
      x    [1,1,n*n,512] bf16 TILE, L1-sharded
      out  [1,1,n*n,2048] bf16 TILE, L1 interleaved, same channel order (stock: no add(0, bfp8) at these sizes)

Replaces S2I -> 3 x (I2S -> halo -> maxpool -> S2I) -> concat -> tilize [-> add(0, bfp8)] -> S2I -> concat.
16 cores, core j owns channels 32j..32j+31 of all positions, so the whole window is core-local:
reader NoC-gathers the input tiles (and copies them to the pass-through part of the output), compute does the
cascade max5 -> max9 -> max13 on the SFPU (kernels/compute_spp.cpp, kernels/compute_spp_g.cpp), writer stores the
pooled tiles. Kernel sources are located relative to this file.
"""
import os

import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_R = open(os.path.join(_HERE, "kernels", "reader_spp.cpp")).read()
_SRC_W = open(os.path.join(_HERE, "kernels", "writer_spp.cpp")).read()
_SRC_C = open(os.path.join(_HERE, "kernels", "compute_spp.cpp")).read()
_SRC_RG = open(os.path.join(_HERE, "kernels", "reader_spp_g.cpp")).read()
_SRC_WG = open(os.path.join(_HERE, "kernels", "writer_spp_g.cpp")).read()
_SRC_CG = open(os.path.join(_HERE, "kernels", "compute_spp_g.cpp")).read()
TILE = 1088  # bfp8_b tile bytes
TILE_BF16 = 2048
MAX_TN = 13  # DST (full sync) holds 16 bf16 tiles; one image tile per 32 positions
H = W = 10
C_IN = 512
N_CORES = C_IN // 32
_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))})
_PLANS = {}


def supported(x):
    if x.layout != ttnn.TILE_LAYOUT or x.dtype != ttnn.bfloat8_b or not x.is_sharded():
        return False
    if tuple(x.shape) != (1, 1, H * W, C_IN) or x.padded_shape[-2] != 128:
        return False
    mc = x.memory_config()
    if mc.buffer_type != ttnn.BufferType.L1:
        return False
    sh, sw = mc.shard_spec.shape
    return sh % 32 == 0 and sw % 32 == 0


class _Shards:
    """Which core holds tile (tr, tc) of a tile-sharded tensor, and at which byte offset of its shard."""

    def __init__(self, mc, tile=TILE):
        sp = mc.shard_spec
        self.tile = tile
        self.sh, self.sw = sp.shape
        self.layout = mc.memory_layout
        self.rm = sp.orientation == ttnn.ShardOrientation.ROW_MAJOR
        if self.layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            rs = list(sp.grid.ranges())
            self.x0 = min(r.start.x for r in rs)
            self.y0 = min(r.start.y for r in rs)
        else:
            self.cores = ttnn.corerange_to_cores(sp.grid, row_wise=self.rm)

    def owner(self, tr, tc):
        r, c = tr * 32, tc * 32
        off = (((r % self.sh) // 32) * (self.sw // 32) + (c % self.sw) // 32) * self.tile
        if self.layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            core = self.cores[r // self.sh]
            return (core.x, core.y), off
        if self.layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            core = self.cores[c // self.sw]
            return (core.x, core.y), off
        i, j = r // self.sh, c // self.sw
        return ((self.x0 + j, self.y0 + i) if self.rm else (self.x0 + i, self.y0 + j)), off


def _plan(dev, mc):
    sh = _Shards(mc)
    cores = [(c.x, c.y) for c in ttnn.corerange_to_cores(_GRID, row_wise=True)]
    per_core = {}
    for j, core in enumerate(cores):
        src = []
        for tr in range(4):
            (ox, oy), off = sh.owner(tr, j)
            v = dev.worker_core_from_logical_core(ttnn.CoreCoord(ox, oy))
            src += [v.x | (v.y << 16), off]
        per_core[core] = (j, src)
    return per_core


def spp(x, mode=0):
    """mode 0 = SPP, 1 = layout round trip (all pooled parts == x; test only)."""
    dev = x.device()
    key = str(x.memory_config())
    if key not in _PLANS:
        _PLANS[key] = _plan(dev, x.memory_config())
    per_core = _PLANS[key]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, H * W, 4 * C_IN]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT,
                                         dev, ttnn.L1_MEMORY_CONFIG)
    oa, xa = out.buffer_address(), x.buffer_address()
    rt_r, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for (cx, cy), (j, src) in per_core.items():
        rt_r[cx][cy] = [oa, j] + [v + xa if k % 2 else v for k, v in enumerate(src)]
        rt_w[cx][cy] = [oa, j]
    ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    cbs = [ttnn.CBDescriptor(total_size=n * TILE, core_ranges=_GRID,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat8_b,
                                                                         page_size=TILE)])
           for i, n in ((0, 4), (16, 12))]
    reader = ttnn.KernelDescriptor(kernel_source=_SRC_R, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                   core_ranges=_GRID, compile_time_args=ct, runtime_args=rt_r,
                                   config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=_SRC_W, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                   core_ranges=_GRID, compile_time_args=ct, runtime_args=rt_w,
                                   config=ttnn.WriterConfigDescriptor())
    ccfg = ttnn.ComputeConfigDescriptor()
    ccfg.fp32_dest_acc_en = False
    ccfg.dst_full_sync_en = True
    ccfg.math_approx_mode = False
    compute = ttnn.KernelDescriptor(kernel_source=_SRC_C, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                    core_ranges=_GRID, compile_time_args=[], defines=[("SPP_MODE", str(mode))],
                                    runtime_args=[], config=ccfg)
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    return ttnn.generic_op([x, out], pd)


# ------------------------------------------------------------------------------------------- generic bf16 path
def supported_g(x, n):
    """384/512/640 path: [1,1,n*n,512] bf16 TILE L1-sharded, tile-aligned shards, n even, n*n <= MAX_TN*32."""
    if n <= 0 or n % 2 or -(-n * n // 32) > MAX_TN:
        return False
    if x.layout != ttnn.TILE_LAYOUT or x.dtype != ttnn.bfloat16 or not x.is_sharded():
        return False
    if tuple(x.shape) != (1, 1, n * n, C_IN) or x.padded_shape[-2] != 32 * -(-n * n // 32):
        return False
    mc = x.memory_config()
    if mc.buffer_type != ttnn.BufferType.L1 or mc.shard_spec is None:
        return False
    sh, sw = mc.shard_spec.shape
    return sh % 32 == 0 and sw % 32 == 0


def _plan_g(dev, mc, tn):
    sh = _Shards(mc, TILE_BF16)
    cores = [(c.x, c.y) for c in ttnn.corerange_to_cores(_GRID, row_wise=True)]
    per_core = {}
    for j, core in enumerate(cores):
        src = []
        for tr in range(tn):
            (ox, oy), off = sh.owner(tr, j)
            v = dev.worker_core_from_logical_core(ttnn.CoreCoord(ox, oy))
            src += [v.x | (v.y << 16), off]
        per_core[core] = (j, src)
    return per_core


def spp_g(x, n, mode=0):
    """mode 0 = SPP, 1 = layout round trip (all pooled parts == x; test only)."""
    dev = x.device()
    tn = -(-n * n // 32)
    key = ("g", str(x.memory_config()), n)
    if key not in _PLANS:
        _PLANS[key] = _plan_g(dev, x.memory_config(), tn)
    per_core = _PLANS[key]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, n * n, 4 * C_IN]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                         dev, ttnn.L1_MEMORY_CONFIG)
    oa, xa = out.buffer_address(), x.buffer_address()
    rt_r, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for (cx, cy), (j, src) in per_core.items():
        rt_r[cx][cy] = [oa, j] + [v + xa if k % 2 else v for k, v in enumerate(src)]
        rt_w[cx][cy] = [oa, j]
    ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    cbs = [ttnn.CBDescriptor(total_size=nt * TILE_BF16, core_ranges=_GRID,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16,
                                                                         page_size=TILE_BF16)])
           for i, nt in ((0, tn), (1, tn), (2, tn), (16, 8))]
    defs = [("N", str(n)), ("TN", str(tn))]
    reader = ttnn.KernelDescriptor(kernel_source=_SRC_RG, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                   core_ranges=_GRID, compile_time_args=ct, defines=defs, runtime_args=rt_r,
                                   config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=_SRC_WG, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                   core_ranges=_GRID, compile_time_args=ct, defines=defs, runtime_args=rt_w,
                                   config=ttnn.WriterConfigDescriptor())
    ccfg = ttnn.ComputeConfigDescriptor()
    ccfg.fp32_dest_acc_en = False
    ccfg.dst_full_sync_en = True
    ccfg.math_approx_mode = False
    compute = ttnn.KernelDescriptor(kernel_source=_SRC_CG, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                    core_ranges=_GRID, compile_time_args=[],
                                    defines=defs + [("SPP_MODE", str(mode))], runtime_args=[], config=ccfg)
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    return ttnn.generic_op([x, out], pd)
