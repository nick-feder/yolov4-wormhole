"""Fused nearest-2x upsample + channel concat (ttnn.generic_op, pure data movement), all neck sizes.

    out = upcat(a, b, h, w, out_mc)        320 graph (bfp8_b): byte-exact == concat([b, up2x(a)]), written
                                           straight into the sharded layout conv7_3 / conv9_3 consume (out_mc)
    out = upcat_g(a, b, h, w)              384/512/640 graphs (bf16): byte-exact == concat([b, up2x(a)], dim=3)
                                           as the stock L1-INTERLEAVED TILE tensor [1,1,4hw,Cb+Ca]
      a  [1,1,h*w,Ca]  TILE, L1-sharded (low-res branch: conv7 / conv9 output)
      b  [1,1,4hw,Cb]  TILE, L1-sharded (lateral branch: conv7_2 / conv9_2 output)

Each output tile is either a lateral tile (whole-tile NoC copy) or an "up" tile gathered pixel row by pixel row from
the needed low-res tiles (staged in a scratch CB), see kernels/upcat_dm_v3.cpp (bfp8) and kernels/upcat_g.cpp (bf16).
Kernel sources are located relative to this file (no absolute paths).
"""
import math
import os
from collections import defaultdict

import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_V3 = open(os.path.join(_HERE, "kernels", "upcat_dm_v3.cpp")).read()
_SRC_G = open(os.path.join(_HERE, "kernels", "upcat_g.cpp")).read()
TILE = 1088  # bfp8_b
TILE_BF16 = 2048
NB_G = 4  # build slots per RISC in upcat_g.cpp
_PLANS = {}
BALANCE = os.environ.get("UPCAT_BAL", "1") != "0"
C_LAT, C_UP, C_REMOTE = 1, 8, 9  # relative RISC cost: tile NoC copy, local row gather, gather + NoC write


def supported(t):
    """320 path: bfp8 TILE L1-sharded, tile-aligned shards."""
    if t.layout != ttnn.TILE_LAYOUT or t.dtype != ttnn.bfloat8_b or not t.is_sharded():
        return False
    mc = t.memory_config()
    if mc.buffer_type != ttnn.BufferType.L1:
        return False
    sh, sw = mc.shard_spec.shape
    return sh % 32 == 0 and sw % 32 == 0


def supported_g(t):
    """generic path: bf16 TILE L1-sharded (height/block/width), tile-aligned shards, channels % 32 == 0."""
    if t.layout != ttnn.TILE_LAYOUT or t.dtype != ttnn.bfloat16 or not t.is_sharded():
        return False
    mc = t.memory_config()
    if mc.buffer_type != ttnn.BufferType.L1 or mc.shard_spec is None:
        return False
    sh, sw = mc.shard_spec.shape
    return sh % 32 == 0 and sw % 32 == 0 and t.shape[-1] % 32 == 0


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


def _up_rows(tr, h, w):
    """Low-res (tile row, row) of each of the 32 rows of output tile row tr (None past 4hw)."""
    w2, n_rows = 2 * w, 4 * h * w
    rows = []
    for i in range(32):
        r = tr * 32 + i
        if r >= n_rows:
            rows.append(None)
        else:
            y, x = divmod(r, w2)
            s = (y // 2) * w + x // 2
            rows.append((s // 32, s % 32))
    return rows


# ------------------------------------------------------------------------------------------- 320 path (verbatim)
def _plan(dev, a_mc, b_mc, out_mc, h, w, ca, cb):
    """Per-core, per-RISC argument lists (without buffer addresses)."""
    sa, sb, so = _Shards(a_mc), _Shards(b_mc), _Shards(out_mc)
    n_rows = 4 * h * w
    ctb, ct = cb // 32, (cb + ca) // 32
    work = defaultdict(lambda: {"lat": [], "up": []})
    noc = {}

    def xy(core):
        if core not in noc:
            v = dev.worker_core_from_logical_core(ttnn.CoreCoord(*core))
            noc[core] = v.x | (v.y << 16)
        return noc[core]

    for tr in range(math.ceil(n_rows / 32)):
        for tc in range(ct):
            core, doff = so.owner(tr, tc)
            if tc < ctb:
                bcore, boff = sb.owner(tr, tc)
                work[core]["lat"].append((xy(bcore), boff, doff))
                continue
            work[core]["up"].append((doff, tc - ctb, _up_rows(tr, h, w)))

    cores = so_cores(out_mc)
    risc = {(c, k): {"lat": [], "up": [], "cost": 0} for c in cores for k in range(2)}
    for core in cores:
        for l_ in work.get(core, {"lat": []})["lat"]:
            q = min(((core, k) for k in range(2)), key=lambda q: risc[q]["cost"])
            risc[q]["lat"].append(l_)
            risc[q]["cost"] += C_LAT
    for core in cores:
        for doff, tca, rows in work.get(core, {"up": []})["up"]:
            own = min(((core, k) for k in range(2)), key=lambda q: risc[q]["cost"])
            best = min(risc, key=lambda q: risc[q]["cost"]) if BALANCE else own
            if best[0] != core and risc[best]["cost"] + C_REMOTE < risc[own]["cost"] + C_UP:
                risc[best]["up"].append((xy(core), doff, tca, rows))
                risc[best]["cost"] += C_REMOTE
            else:
                risc[own]["up"].append((0, doff, tca, rows))
                risc[own]["cost"] += C_UP

    per_core = {}
    max_slots = 1
    for core in cores:
        args, slot0 = [], 0
        for k in range(2):
            rk = risc[(core, k)]
            slots = {}
            ups = []
            for dxy, doff, tca, rows in rk["up"]:
                ents = []
                for rr in rows:
                    if rr is None:
                        ents.append(0xFFFF)
                        continue
                    key = (rr[0], tca)
                    if key not in slots:
                        slots[key] = len(slots)
                    ents.append((slots[key] << 5) | rr[1])
                ups.append([dxy, doff] + [ents[2 * p] | (ents[2 * p + 1] << 16) for p in range(16)])
            src = []
            for (ta, tca), _ in sorted(slots.items(), key=lambda kv: kv[1]):
                acore, aoff = sa.owner(ta, tca)
                src += [xy(acore), aoff]
            lst = [slot0, len(slots), len(rk["lat"]), len(ups)] + src
            for l_ in rk["lat"]:
                lst += list(l_)
            for u in ups:
                lst += u
            args.append(lst)
            slot0 += len(slots) + sum(1 for u in rk["up"] if u[0])  # source slots + build slots
        max_slots = max(max_slots, slot0)
        per_core[core] = args
    return per_core, max_slots


def so_cores(mc):
    return [(c.x, c.y) for c in ttnn.corerange_to_cores(mc.shard_spec.grid, row_wise=True)]


def upcat(a, b, h, w, out_mc):
    ca, cb = a.shape[-1], b.shape[-1]
    n_rows = 4 * h * w
    dev = a.device()
    key = ("v3", str(a.memory_config()), str(b.memory_config()), str(out_mc), h, w, ca, cb, BALANCE)
    if key not in _PLANS:
        _PLANS[key] = _plan(dev, a.memory_config(), b.memory_config(), out_mc, h, w, ca, cb)
    per_core, max_slots = _PLANS[key]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, n_rows, ca + cb]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT,
                                         dev, out_mc)
    base = [out.buffer_address(), a.buffer_address(), b.buffer_address()]
    rts = [ttnn.RuntimeArgs(), ttnn.RuntimeArgs()]
    for (x, y), args in per_core.items():
        for k in range(2):
            rts[k][x][y] = base + args[k]
    grid = out_mc.shard_spec.grid
    cbs = [ttnn.CBDescriptor(total_size=max_slots * TILE, core_ranges=grid,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0,
                                                                         data_format=ttnn.bfloat8_b,
                                                                         page_size=TILE)])]
    kernels = [ttnn.KernelDescriptor(kernel_source=_SRC_V3, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                     core_ranges=grid, compile_time_args=[], runtime_args=rts[k], config=cfg)
               for k, cfg in enumerate([ttnn.WriterConfigDescriptor(), ttnn.ReaderConfigDescriptor()])]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op([a, b, out], pd)


# ------------------------------------------------------------------------------------------- generic bf16 path
def _plan_g(dev, a_mc, b_mc, h, w, ca, cb):
    """Lateral tiles: written by the core owning them (NoC write straight from its shard). Up tiles: spread in
    (column, row) order over all RISCs of the worker grid so neighbours share their low-res source tiles."""
    sa, sb = _Shards(a_mc, TILE_BF16), _Shards(b_mc, TILE_BF16)
    n_tr = math.ceil(4 * h * w / 32)
    ctb, ct = cb // 32, (cb + ca) // 32
    gs = dev.compute_with_storage_grid_size()
    cores = [(x, y) for y in range(gs.y) for x in range(gs.x)]
    noc = {}

    def xy(core):
        if core not in noc:
            v = dev.worker_core_from_logical_core(ttnn.CoreCoord(*core))
            noc[core] = v.x | (v.y << 16)
        return noc[core]

    risc = {(c, k): {"lat": [], "up": [], "cost": 0} for c in cores for k in range(2)}
    for tr in range(n_tr):
        for tc in range(ctb):
            bcore, boff = sb.owner(tr, tc)
            q = min(((bcore, k) for k in range(2)), key=lambda q: risc[q]["cost"])
            risc[q]["lat"].append((boff, tr * ct + tc))
            risc[q]["cost"] += C_LAT
    ups = [(tc, tr) for tc in range(ctb, ct) for tr in range(n_tr)]
    total = sum(r["cost"] for r in risc.values()) + C_UP * len(ups)
    order = [(c, k) for k in range(2) for c in cores]
    target = total / len(order)
    i = 0
    for q in order:
        while i < len(ups) and (risc[q]["cost"] + C_UP / 2 <= target or q == order[-1]):
            risc[q]["up"].append(ups[i])
            risc[q]["cost"] += C_UP
            i += 1
    assert i == len(ups)

    per_core = {}
    max_slots = 1
    for core in cores:
        args, slot0 = [], 0
        for k in range(2):
            rk = risc[(core, k)]
            slots, recs = {}, []
            for tc, tr in rk["up"]:
                ents = []
                for rr in _up_rows(tr, h, w):
                    if rr is None:
                        ents.append(0xFFFF)
                        continue
                    key = (rr[0], tc - ctb)
                    if key not in slots:
                        slots[key] = len(slots)
                    ents.append((slots[key] << 5) | rr[1])
                recs.append([tr * ct + tc] + [ents[2 * p] | (ents[2 * p + 1] << 16) for p in range(16)])
            src = []
            for (ta, tca), _ in sorted(slots.items(), key=lambda kv: kv[1]):
                acore, aoff = sa.owner(ta, tca)
                src += [xy(acore), aoff]
            lst = [slot0, len(slots), len(rk["lat"]), len(recs)] + src
            for l_ in rk["lat"]:
                lst += list(l_)
            for r_ in recs:
                lst += r_
            args.append(lst)
            slot0 += len(slots) + (NB_G if recs else 0)
        max_slots = max(max_slots, slot0)
        per_core[core] = args
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gs.x - 1, gs.y - 1))})
    return per_core, max_slots, grid


def upcat_g(a, b, h, w, memory_config=None):
    ca, cb = a.shape[-1], b.shape[-1]
    n_rows = 4 * h * w
    dev = a.device()
    key = ("g", str(a.memory_config()), str(b.memory_config()), h, w, ca, cb)
    if key not in _PLANS:
        _PLANS[key] = _plan_g(dev, a.memory_config(), b.memory_config(), h, w, ca, cb) + (len(_PLANS),)
    per_core, max_slots, grid, pid = _PLANS[key]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, n_rows, ca + cb]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                         dev, memory_config or ttnn.L1_MEMORY_CONFIG)
    base = [out.buffer_address(), a.buffer_address(), b.buffer_address()]
    rts = [ttnn.RuntimeArgs(), ttnn.RuntimeArgs()]
    for (x, y), args in per_core.items():
        for k in range(2):
            rts[k][x][y] = base + args[k]
    ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    cbs = [ttnn.CBDescriptor(total_size=max_slots * TILE_BF16, core_ranges=grid,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0,
                                                                         data_format=ttnn.bfloat16,
                                                                         page_size=TILE_BF16)])]
    # the plan id define makes the generic_op program hash unique per plan (runtime-arg layout and CB size differ;
    # without it the two chains share a cached program and the runtime args overflow)
    kernels = [ttnn.KernelDescriptor(kernel_source=_SRC_G, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                     core_ranges=grid, compile_time_args=ct, defines=[("UPCAT_PLAN", str(pid))],
                                     runtime_args=rts[k], config=cfg)
               for k, cfg in enumerate([ttnn.WriterConfigDescriptor(), ttnn.ReaderConfigDescriptor()])]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op([a, b, out], pd)
