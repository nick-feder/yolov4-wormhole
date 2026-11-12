"""Composable patch: YOLOv4 neck "upsample(2x) -> concat" chains as ONE custom generic_op each, all input sizes.

    patch_upcat.apply()          # after the combo folder's submission.py (patch_cfg) is loaded, BEFORE patch_spp

Rebinds TtNeck.__call__ (models/demos/yolov4/tt/neck.py, the shadow copy of the loaded combo) to `_neck_call`, a
verbatim copy of the stock __call__ (both the 320 and the 384/512/640 branches) in which the two chains
    conv7 out -> untilize -> reshape -> reshard -> upsample -> s2i -> reshape -> tilize [-> add(0,bfp8)] -+-> concat
    conv7_2 out -> s2i -----------------------------------------------------------------------------------+
(and conv9 / conv9_2) are replaced by cat1() / cat2():
  320          upcat.upcat()   (bfp8, written straight into the sharded layout conv7_3 / conv9_3 consume)
  384/512/640  upcat.upcat_g() (bf16, the stock [1,1,4hw,Cb+Ca] TILE L1-interleaved concat output)
Both are byte-exact (pure data movement). Anything unsupported runs the verbatim stock chain (STATS["fallback"]).

Env: FB_UPCAT=1 (0 = stock chains through the same __call__ copy).
Stacking: apply() re-points this module's `ttnn` global to the neck module's `ttnn` (patch_cfg's `_Proxy`, if one
is installed), so patch_cfg's toggles stay active inside the copied __call__. patch_spp.apply() must come after
(it proxies the globals of whatever TtNeck.__call__ is current, i.e. this module).
"""
import importlib.util
import os
import sys

import ttnn
from models.demos.yolov4.tt import neck as _neck

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


_uc = _load_sibling("upcat")

ENABLED = os.environ.get("FB_UPCAT", "1") != "0"
STATS = {"custom": 0, "fallback": 0}
_FAILED = set()
HOOK = None  # test hook: HOOK(name, tensor, neck)
_ORIG_CALL = _neck.TtNeck.__call__


def _mc_bs(nx, ny, sh, sw):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))})
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(grid, (sh, sw), ttnn.ShardOrientation.ROW_MAJOR))


def _mc_hs(n, sh, sw):
    grid = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(8, 8), True)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(grid, (sh, sw), ttnn.ShardOrientation.ROW_MAJOR))


# 320: input layouts the stock conv7_3 / conv9_3 reshard to (448x512 BS 56 cores, 1600x256 HS 50 cores).
OUT_MC = {1: lambda: _mc_bs(8, 7, 64, 64), 2: lambda: _mc_hs(50, 32, 256)}


def _hook(name, t, neck):
    if HOOK is not None:
        HOOK(name, t, neck)


# ---------------------------------------------------------------- stock chains (verbatim from neck.py)
def _untilize_reshape(self, which, output_tensor):
    c = self.conv_args.c7 if which == 1 else self.conv_args.c9
    output_shape = output_tensor.shape
    output_tensor = ttnn.untilize_with_unpadding(
        output_tensor,
        output_tensor_end=(output_shape[0] - 1, output_shape[1] - 1, output_shape[2] - 1, output_shape[3] - 1),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return ttnn.reshape(output_tensor, (1, c.input_height, c.input_width, c.out_channels))


def _upsample_320(which, output_tensor):
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))})
    sh, sw = (20, 32) if which == 1 else (80, 16)
    shard_spec = ttnn.ShardSpec(shard_grid, (sh, sw), ttnn.ShardOrientation.ROW_MAJOR)
    in_sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)
    output_tensor = ttnn.to_memory_config(output_tensor, memory_config=in_sharded_mem_config)
    shard_spec = ttnn.ShardSpec(shard_grid, (sh * 4, sw), ttnn.ShardOrientation.ROW_MAJOR)
    out_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec)
    up = ttnn.upsample(output_tensor, (2, 2), memory_config=out_sharded_mem_config)
    return ttnn.sharded_to_interleaved(up, ttnn.L1_MEMORY_CONFIG)


def _upsample_g(output_tensor):
    nhw = output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2]
    num_cores = _neck.determine_num_cores_for_upsample(nhw, output_tensor.shape[2])
    core_grid = _neck.get_core_grid_from_num_cores(num_cores)
    shardspec = ttnn.create_sharded_memory_config_(
        output_tensor.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    if _neck.is_blackhole():
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    else:
        output_tensor = ttnn.to_memory_config(output_tensor, shardspec)
    up = ttnn.upsample(output_tensor, scale_factor=2, memory_config=output_tensor.memory_config())
    return ttnn.sharded_to_interleaved(up, ttnn.L1_MEMORY_CONFIG)


def stock_post(self, which, output_tensor, lateral):
    """Verbatim stock chain (both branches) from conv7/conv9 output + conv7_2/conv9_2 output to the concat."""
    is320 = self.parameters.resolution[0] == 320
    output_tensor = _untilize_reshape(self, which, output_tensor)
    up = _upsample_320(which, output_tensor) if is320 else _upsample_g(output_tensor)
    up = ttnn.reshape(up, (1, 1, up.shape[1] * up.shape[2], up.shape[3]))
    up = ttnn.to_layout(up, layout=ttnn.TILE_LAYOUT)
    lateral = ttnn.sharded_to_interleaved(lateral, ttnn.L1_MEMORY_CONFIG)
    if is320:
        up = ttnn.add(up, 0.0, dtype=ttnn.bfloat8_b)
    out = ttnn.concat([lateral, up], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(up)
    return out


# ---------------------------------------------------------------- custom chains
def _mode(self, low, lateral):
    """'320' / 'g' (custom kernel) or None (stock)."""
    if not ENABLED:
        return None
    if self.parameters.resolution[0] == 320:
        return "320" if _uc.supported(low) and _uc.supported(lateral) else None
    if _uc.supported_g(low) and _uc.supported_g(lateral) and low.dtype == lateral.dtype:
        return "g"
    return None


def _cat(self, which, low, lateral):
    _hook(f"a{which}", low, self)
    _hook(f"b{which}", lateral, self)
    c = self.conv_args.c7 if which == 1 else self.conv_args.c9
    mode = _mode(self, low, lateral)
    if mode is not None and (low.shape[-2] != c.input_height * c.input_width
                             or lateral.shape[-2] != 4 * c.input_height * c.input_width):
        mode = None
    key = (which, mode, str(low.memory_config()), str(lateral.memory_config()))
    if mode is not None and key in _FAILED:
        mode = None
    try:
        if mode == "320":
            out = _uc.upcat(low, lateral, c.input_height, c.input_width, OUT_MC[which]())
        elif mode == "g":
            out = _uc.upcat_g(low, lateral, c.input_height, c.input_width)
    except Exception as ex:  # e.g. scratch CBs clash with L1 buffers: remember and run stock from now on
        _FAILED.add(key)
        print(f"[patch_upcat] custom chain {which} failed, stock fallback: {str(ex).splitlines()[0][:200]}", flush=True)
        mode = None
    if mode is None:
        STATS["fallback"] += 1
        return stock_post(self, which, low, lateral)
    STATS["custom"] += 1
    ttnn.deallocate(low)
    ttnn.deallocate(lateral)
    return out


def cat1(self, conv7_out, d4):
    """concat([conv7_2(d4), up2x(conv7_out)], dim=3) as conv7_3 consumes it."""
    return _cat(self, 1, conv7_out, self.conv7_2(d4))


def cat2(self, conv9_out, d3):
    """concat([conv9_2(d3), up2x(conv9_out)], dim=3) as conv9_3 consumes it."""
    return _cat(self, 2, conv9_out, self.conv9_2(d3))


# ---------------------------------------------------------------- TtNeck.__call__ copy (all sizes)
def _neck_call(self, input_tensor):
    output_tensor = self.conv1(input_tensor[0])
    output_tensor = self.conv2(output_tensor)
    output_tensor = self.conv3(output_tensor)

    output_tensor_pool_in = output_tensor
    if output_tensor_pool_in.memory_config().is_sharded():
        output_tensor_pool_in = ttnn.sharded_to_interleaved(output_tensor_pool_in, ttnn.L1_MEMORY_CONFIG)
    pools = []
    for p in (self.conv_args.p1, self.conv_args.p2, self.conv_args.p3):
        pools.append(ttnn.max_pool2d(
            input_tensor=output_tensor_pool_in,
            batch_size=p.batch_size,
            input_h=p.input_height,
            input_w=p.input_width,
            channels=output_tensor.shape[3],
            kernel_size=[p.kernel_size, p.kernel_size],
            stride=[p.stride, p.stride],
            padding=[p.padding, p.padding],
            dilation=[p.dilation, p.dilation],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ))
    pool_1, pool_2, pool_3 = pools
    if pool_1.memory_config().is_sharded():
        pool_1 = ttnn.sharded_to_interleaved(pool_1, ttnn.L1_MEMORY_CONFIG)
    if pool_2.memory_config().is_sharded():
        pool_2 = ttnn.sharded_to_interleaved(pool_2, ttnn.L1_MEMORY_CONFIG)
    if pool_3.memory_config().is_sharded():
        pool_3 = ttnn.sharded_to_interleaved(pool_3, ttnn.L1_MEMORY_CONFIG)

    pool_all = ttnn.concat([pool_3, pool_2, pool_1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    pool_all = ttnn.to_layout(pool_all, layout=ttnn.TILE_LAYOUT)
    if self.parameters.resolution[0] == 320:
        pool_all = ttnn.add(pool_all, 0.0, dtype=ttnn.bfloat8_b)
    ttnn.deallocate(pool_3)
    ttnn.deallocate(pool_2)
    ttnn.deallocate(pool_1)

    output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.concat([pool_all, output_tensor], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = self.conv4(output_tensor)
    output_tensor = self.conv5(output_tensor)
    output_tensor_left_1 = self.conv6(output_tensor)
    output_tensor = self.conv7(output_tensor_left_1)

    output_tensor = cat1(self, output_tensor, input_tensor[1])

    output_tensor = self.conv7_3(output_tensor)
    output_tensor = self.conv8(output_tensor)
    output_tensor = self.conv7_4(output_tensor)
    output_tensor = self.conv8_2(output_tensor)
    output_tensor_left_2 = self.conv7_5(output_tensor)

    if self.parameters.resolution[0] == 320:
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 3))})
        shard_spec = ttnn.ShardSpec(shard_grid, (64, 64), ttnn.ShardOrientation.COL_MAJOR)
        in_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)
        output_tensor_left_2 = ttnn.to_memory_config(output_tensor_left_2, memory_config=in_sharded_mem_config)

    output_tensor = self.conv9(output_tensor_left_2)

    output_tensor = cat2(self, output_tensor, input_tensor[2])

    output_tensor = self.conv9_3(output_tensor)
    output_tensor = self.conv10(output_tensor)
    output_tensor = self.conv9_4(output_tensor)
    output_tensor = self.conv10_2(output_tensor)
    output_tensor = self.conv9_5(output_tensor)

    ttnn.deallocate(input_tensor[0])
    ttnn.deallocate(input_tensor[1])
    ttnn.deallocate(input_tensor[2])

    return output_tensor, output_tensor_left_1, output_tensor_left_2


def link_cfg_proxy():
    """Use the neck module's `ttnn` (e.g. patch_cfg's _Proxy) inside the copied __call__ and the stock chains."""
    nt = _neck.ttnn
    if nt is not globals()["ttnn"] and type(nt).__name__ == "_Proxy":
        globals()["ttnn"] = nt
        return True
    return False


def apply():
    linked = link_cfg_proxy()
    _neck.TtNeck.__call__ = _neck_call
    print(f"[patch_upcat] TtNeck.__call__ patched, FB_UPCAT={int(ENABLED)}, cfg proxy linked={linked}", flush=True)
