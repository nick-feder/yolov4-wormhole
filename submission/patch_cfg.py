"""Config-only / Python-only speedups for the final YOLOv4 pipeline (todo arch-cfg). Composable: apply() only
wraps what is already there (TtConv2d.__init__, the `ttnn` global of head/neck/rawheads_model, and
input_stage.create_pipeline_from_config), so other patches applied before or after keep working.

Env toggles (all default off):
    CFG_HALO=1|lo:hi   reallocate_halo_output=False on TtConv2d #lo..hi-1 (1 = all) and on the 3 SPP maxpools
                       -> drops the ttnn.move (L1->L1 defrag copy) after each halo
    CFG_NOINMOVE=1     drop the ttnn.reallocate(nhwc) Move in the input stage (rawheads_model._backbone_heads)
    CFG_NOADD=1        skip `ttnn.add(x, 0.0, dtype=d)` in head/neck when x.dtype is already d (bfp8->bfp8 no-op)
    CFG_IO=1           output read on CQ1: trace on CQ0 -> ttnn.copy(L1 out -> persistent DRAM buffer) -> D2H read
                       on CQ1 with events, so the next frame's trace overlaps the read of the previous output
    CFG_DEEP=k=v,...   Conv2dConfig / Conv2dConfiguration overrides for deep convs (3x3, in_channels >= CFG_DEEP_MINC
                       [512]), e.g. full_inner_dim=1, enable_weights_double_buffer=0, packer_l1_acc=1,
                       act_block_h_override=32
    CFG_LOG=1          print every TtConv2d's config at build time
"""
import dataclasses
import os
import sys

import ttnn

_STATS = {"conv": 0, "halo_off": 0, "pool_halo_off": 0, "noadd_skip": 0, "noadd_kept": 0, "deep": 0,
          "inmove_skip": 0}
_APPLIED = False


def _env(name, default=""):
    return os.environ.get(name, default).strip()


class _Proxy:
    """Stand-in for a module's `ttnn` global; forwards everything except the overridden names."""

    def __init__(self, real, **over):
        self.__dict__["_real"] = real
        self.__dict__.update(over)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _range(spec):
    if spec in ("", "0"):
        return None
    if spec == "1":
        return (0, 10**9)
    lo, hi = spec.split(":")
    return int(lo), int(hi)


def _parse_kv(spec):
    out = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        k, v = item.split("=")
        out[k] = int(v) if v.lstrip("-").isdigit() else v
    return out


def _patch_convs(halo, deep, minc, log):
    from models.tt_cnn.tt import builder

    orig_init = builder.TtConv2d.__init__
    fields = {f.name for f in dataclasses.fields(builder.Conv2dConfiguration)}

    def init(self, configuration, device):
        i = _STATS["conv"]
        _STATS["conv"] += 1
        c = configuration
        post = {}
        if halo and halo[0] <= i < halo[1] and c.reallocate_halo_output:
            c = dataclasses.replace(c, reallocate_halo_output=False)
            _STATS["halo_off"] += 1
        is_deep = deep and tuple(c.kernel_size) == (3, 3) and c.in_channels >= minc
        if is_deep:
            _STATS["deep"] += 1
            rep = {}
            for k, v in deep.items():
                if k in fields and k != "act_block_h_override":
                    rep[k] = bool(v) if isinstance(getattr(c, k), bool) else v
                else:
                    post[k] = v
            c = dataclasses.replace(c, **rep)
        orig_init(self, c, device)
        for k, v in post.items():
            cur = getattr(self.conv2d_config, k)
            setattr(self.conv2d_config, k, bool(v) if isinstance(cur, bool) else v)
        if log:
            cc = self.conv2d_config
            print(f"[patch_cfg] conv#{i} {c.in_channels}->{c.out_channels} k{tuple(c.kernel_size)} s{tuple(c.stride)} "
                  f"in {c.input_height}x{c.input_width} shard={cc.shard_layout} abh={cc.act_block_h_override} "
                  f"dealloc={c.deallocate_activation} realloc_halo={cc.reallocate_halo_output} "
                  f"wdb={cc.enable_weights_double_buffer} adb={cc.enable_act_double_buffer} "
                  f"fid={cc.full_inner_dim} l1acc={c.packer_l1_acc} out={c.output_dtype}"
                  f"{' DEEP' if is_deep else ''}", flush=True)

    builder.TtConv2d.__init__ = init


def _patch_head_neck(noadd, pool_halo):
    import models.demos.yolov4.tt.head as head
    import models.demos.yolov4.tt.neck as neck

    def make_add(real_add):
        def add(a, b, *args, **kw):
            if not args and isinstance(b, (int, float)) and b == 0 and set(kw) == {"dtype"}:
                if getattr(a, "dtype", None) == kw["dtype"]:
                    _STATS["noadd_skip"] += 1
                    return a
                _STATS["noadd_kept"] += 1
            return real_add(a, b, *args, **kw)
        return add

    def make_pool(real_pool):
        def max_pool2d(*args, **kw):
            kw.setdefault("reallocate_halo_output", False)
            _STATS["pool_halo_off"] += 1
            return real_pool(*args, **kw)
        return max_pool2d

    if noadd:
        head.ttnn = _Proxy(head.ttnn, add=make_add(head.ttnn.add))
    over = {}
    if noadd:
        over["add"] = make_add(neck.ttnn.add)
    if pool_halo:
        over["max_pool2d"] = make_pool(neck.ttnn.max_pool2d)
    if over:
        neck.ttnn = _Proxy(neck.ttnn, **over)


def _patch_input_move():
    rh = sys.modules.get("rawheads_model")
    if rh is None:
        raise RuntimeError("patch_cfg CFG_NOINMOVE: rawheads_model not loaded (load submission_rh first)")

    def reallocate(t, *a, **k):
        _STATS["inmove_skip"] += 1
        return t

    rh.ttnn = _Proxy(rh.ttnn, reallocate=reallocate)


def _make_io_executor():
    from models.tt_cnn.tt.executor import MultiCQTracedModelPipelinedIOExecutor as Base

    class OutputReadOnCQ1(Base):
        """MultiCQTracedModelPipelinedIOExecutor with ttnn.copy instead of reshard (our output is an
        L1-interleaved ROW_MAJOR tensor, not sharded) and no >=2-inputs restriction."""

        def _compile_model(self, host_input):
            ttnn.wait_for_event(self.CQ_IO, self.first_op_event)
            ttnn.copy_host_to_device_tensor(host_input, self.dram_input_tensor, cq_id=self.CQ_IO)
            self.write_event = ttnn.record_event(self.device, self.CQ_IO)
            ttnn.wait_for_event(self.CQ_OPS, self.write_event)
            l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)
            self.first_op_event = ttnn.record_event(self.device, self.CQ_OPS)
            self._compilation_output_tensor = self.model(l1_input_tensor)
            self.last_op_event = ttnn.record_event(self.device, self.CQ_OPS)
            ttnn.deallocate(l1_input_tensor)

        def _execute_model_and_transfer_output(self):
            ttnn.wait_for_event(self.CQ_OPS, self.write_event)
            self.l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config,
                                                self.l1_input_tensor)
            self.first_op_event = ttnn.record_event(self.device, self.CQ_OPS)
            ttnn.execute_trace(self.device, self.trace_id, cq_id=self.CQ_OPS, blocking=False)
            ttnn.wait_for_event(self.CQ_OPS, self.read_event)
            ttnn.copy(self.output_tensor, self.dram_output_tensor)
            self.last_op_event = ttnn.record_event(self.device, self.CQ_OPS)

        def execute(self, host_inputs):
            n = len(host_inputs)
            if n == 0:
                return
            self._transfer_input_to_device(host_inputs[0])
            for i in range(1, n):
                self._execute_model_and_transfer_output()
                self._transfer_input_to_device(host_inputs[i])
                yield self._read_output_from_device()
            self._execute_model_and_transfer_output()
            yield self._read_output_from_device()

    return OutputReadOnCQ1


def _patch_io():
    stage = sys.modules.get("input_stage")
    if stage is None:
        raise RuntimeError("patch_cfg CFG_IO: input_stage not loaded (load submission_rh first)")
    from models.tt_cnn.tt.pipeline import Pipeline

    orig = stage.create_pipeline_from_config
    exec_cls = _make_io_executor()

    def create(config, model, device, l1_input_memory_config, dram_input_memory_config=None,
               dram_output_memory_config=None):
        if not (config.use_trace and config.num_command_queues == 2 and dram_input_memory_config is not None):
            return orig(config, model, device, l1_input_memory_config, dram_input_memory_config,
                        dram_output_memory_config)
        print("[patch_cfg] CFG_IO: output read on CQ1 (OutputReadOnCQ1 executor)", flush=True)
        return Pipeline(exec_cls(model, device, dram_input_memory_config=dram_input_memory_config,
                                 l1_input_memory_config=l1_input_memory_config,
                                 dram_output_memory_config=dram_output_memory_config or ttnn.DRAM_MEMORY_CONFIG))

    stage.create_pipeline_from_config = create


def apply():
    global _APPLIED
    if _APPLIED:
        return _STATS
    _APPLIED = True
    halo = _range(_env("CFG_HALO"))
    deep = _parse_kv(_env("CFG_DEEP"))
    minc = int(_env("CFG_DEEP_MINC", "512"))
    log = _env("CFG_LOG") == "1"
    noadd = _env("CFG_NOADD") == "1"
    if halo or deep or log:
        _patch_convs(halo, deep, minc, log)
    if noadd or halo:
        _patch_head_neck(noadd, pool_halo=bool(halo))
    if _env("CFG_NOINMOVE") == "1":
        _patch_input_move()
    if _env("CFG_IO") == "1":
        _patch_io()
    print(f"[patch_cfg] HALO={_env('CFG_HALO') or 0} NOINMOVE={_env('CFG_NOINMOVE') or 0} NOADD={int(noadd)} "
          f"IO={_env('CFG_IO') or 0} DEEP={deep or 0} MINC={minc}", flush=True)
    return _STATS
