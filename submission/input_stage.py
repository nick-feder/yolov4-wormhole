# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The same detector on the fast path, driven through the submission interface.

submission_reference.py is the plain path and the thing you are asked to beat.
This is the other end of the same model: the graph captured once and replayed,
a second command queue, and the host decoding the next photographs while the
card works on the current ones.

It exists for two reasons. For the organisers it answers a question the plain
baseline cannot: what the scoring tool reports when the documented techniques
are actually used. For a team it is a worked example of the three things that
matter together, because each one alone leaves most of the win on the table:

    1. ask for the device you need      DEVICE_OPTIONS, below
    2. capture the graph once           create_pipeline_from_config(use_trace=True)
    3. keep the host out of the way     decode ahead, on other threads

Measured figures live in ../data/ and in the README. Nothing here is tuned
beyond those three ideas: there is deliberately more left for a team to find.

    python harness/evaluate.py --submission harness/submission_reference_fast.py \
      --images practice/images --annotations practice/instances.json \
      --team as-supplied-fast --out as-supplied-fast.json
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
import ttnn

from models.demos.yolov4.common import get_model_result, image_to_tensor
from models.demos.yolov4.post_processing import post_processing
from models.demos.yolov4.runner.performant_runner_infra import YOLOv4PerformanceRunnerInfra
from models.demos.yolov4.runner.pipeline_runner import YoloV4PipelineRunner
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

try:
    from models.demos.utils.common_demo_utils import get_mesh_mappers
except Exception:                                     # older trees
    get_mesh_mappers = None

import submission_api as api

RESOLUTION = (320, 320)
CONF_THRESH = 0.001        # the same floor as the plain reference, for comparability
NMS_THRESH = 0.4

# Enough trace room for this model at 320x320; the value bench_levers.py uses.
TRACE_REGION_SIZE = 6434816

# How many frames are handed to the card at a time, and how many threads decode
# ahead of it. Both are worth an experiment: too small and the card waits for
# the host, too large and you hold a lot of decoded frames in memory for nothing.
# Input path v2 (chanyoung): thread knobs from the environment; unset = library default.
CHUNK = int(os.environ.get('CHUNK', 32))
DECODERS = int(os.environ.get('DECODERS', 8))
CV2_THREADS = os.environ.get('CV2_THREADS')
TORCH_THREADS = os.environ.get('TORCH_THREADS', '1')  # sweep 10 Nov: 1 beats default 48 by ~1.5 ms/frame
WAIT_PROFILE = os.environ.get('WAIT_PROFILE') == '1'
if CV2_THREADS is not None:
    cv2.setNumThreads(int(CV2_THREADS))
if TORCH_THREADS is not None:
    torch.set_num_threads(int(TORCH_THREADS))


class Submission(api.Submission):

    # Without this the harness opens the card with no trace region and one
    # command queue, and setup() below fails. Neither can be changed after the
    # device is open, which is why it is declared here rather than requested
    # later.
    DEVICE_OPTIONS = {"trace_region_size": TRACE_REGION_SIZE,
                      "num_command_queues": 2}

    def setup(self, device):
        """Build the model and capture the graph. Not timed."""
        if device is None:
            raise RuntimeError("this needs a card; the harness could not open one")
        self.device = device
        self.resolution = RESOLUTION
        if get_mesh_mappers is not None:
            self.in_mapper, _, self.out_composer = get_mesh_mappers(device)
        else:
            self.in_mapper = self.out_composer = None

        self.infra = YOLOv4PerformanceRunnerInfra(
            device, 1, ttnn.bfloat8_b, ttnn.bfloat8_b,
            model_location_generator=None, resolution=self.resolution,
            mesh_mapper=self.in_mapper, mesh_composer=self.out_composer)

        self.seed_host, dram_cfg, l1_cfg = self.infra.setup_dram_sharded_input(device)
        self.pipeline = create_pipeline_from_config(
            config=PipelineConfig(use_trace=True, num_command_queues=2,
                                  all_transfers_on_separate_command_queue=False),
            model=YoloV4PipelineRunner(self.infra), device=device,
            dram_input_memory_config=dram_cfg, l1_input_memory_config=l1_cfg)
        self.pipeline.compile(self.seed_host)
        self.pool = ThreadPoolExecutor(max_workers=DECODERS)

    def warmup(self, paths):
        """Settle the card and let the kernel cache fill. Not timed."""
        self.run(paths)

    def run(self, paths):
        """TIMED. Decode ahead on the host, replay the graph on the card."""
        paths = list(paths)
        out = {}
        chunks = [paths[i:i + CHUNK] for i in range(0, len(paths), CHUNK)]
        if not chunks:
            return out

        pending = self._submit(chunks[0])
        for i, chunk in enumerate(chunks):
            # Start decoding the next chunk before asking the card for this one,
            # so the two happen at the same time rather than one after the other.
            nxt = self._submit(chunks[i + 1]) if i + 1 < len(chunks) else None
            tw = time.perf_counter()
            frames = [f.result() for f in pending]
            if WAIT_PROFILE:
                self.wait_s = getattr(self, 'wait_s', 0.0) + time.perf_counter() - tw
            pending = nxt

            hosts = [t for t, _, _ in frames]   # already ttnn host tensors (built in _decode)
            results = self.pipeline.enqueue(hosts).pop_all()

            for path, (_, W, H), raw in zip(chunk, frames, results):
                out[path] = self._detections(raw, W, H)
        return out

    # ------------------------------------------------------------------ host
    def _submit(self, chunk):
        return [self.pool.submit(self._decode, p) for p in chunk]

    def _decode(self, path):
        """Runs on a worker thread. cv2 and torch release the interpreter lock,
        so this genuinely overlaps with the card."""
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            return self._to_host(torch.zeros(1, 3, self.resolution[1], self.resolution[0], dtype=torch.bfloat16)), 1, 1
        H, W = frame.shape[:2]
        img = cv2.resize(frame, self.resolution)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Input path v1 (chanyoung): bit-identical to image_to_tensor()+from_torch(bf16), but
        # converted to a contiguous bf16 tensor first. from_torch on the old strided float32
        # tensor cost 1.40 ms/frame (and holds the GIL); on contiguous bf16 it costs 0.07 ms.
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).contiguous().div_(255.0)
        return self._to_host(t), W, H

    def _to_host(self, t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                               mesh_mapper=self.in_mapper)

    def _detections(self, raw, W, H):
        """Identical conversions to submission_reference.py, so the two scores
        are comparable and any difference is speed rather than arithmetic."""
        # The pipeline hands back the model's raw ttnn tensors. YOLOv4Runner
        # puts them through get_model_result before returning, which unpads the
        # boxes and converts both to torch; post_processing expects that form.
        boxes, confs = get_model_result(raw, self.resolution,
                                        mesh_composer=self.out_composer)
        result = post_processing(None, CONF_THRESH, NMS_THRESH, [boxes, confs])
        dets = []
        for det in (result[0] if result else []):
            x1, y1, x2, y2 = (float(det[0]), float(det[1]), float(det[2]), float(det[3]))
            score = float(det[4]) * float(det[5])
            cls_id = int(det[6])
            px1 = min(max(x1 * W, 0.0), W)
            py1 = min(max(y1 * H, 0.0), H)
            px2 = min(max(x2 * W, 0.0), W)
            py2 = min(max(y2 * H, 0.0), H)
            if px2 <= px1 or py2 <= py1:
                continue
            dets.append({"category_id": api.to_coco_id(cls_id),
                         "bbox": api.xyxy_to_xywh(px1, py1, px2, py2),
                         "score": score})
        return dets

    def teardown(self):
        if WAIT_PROFILE:
            print(f'WAIT_PROFILE main thread waited on decoders: {getattr(self, "wait_s", 0.0):.3f} s total', flush=True)
        pool = getattr(self, "pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            try:
                pipeline.cleanup()
            except Exception:
                pass
