"""TEAM SUBMISSION (best so far, 10 Nov: 5.90 ms/frame, mAP 0.464, dets identical to reference 200/200).
Input path (chanyoung, input_stage.py: bf16 + from_torch in decoder threads,
TORCH/DECODERS/CHUNK knobs) + Nick's output stage (box step on worker threads overlapping the next chunk).

    POST=raw (default)      : fast_post2.detections_from_raw (no get_model_result)
    POST=fast_post          : get_model_result + fast_post.detections (7.42 ms variant)
    POST_WORKERS (default 2), DUMP_DETS=/path.pkl to check exactness with nick/cmp_dets.py
"""
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fast_post  # noqa: E402  (copy of nick/fast_post.py)
import fast_post2  # noqa: E402  (copy of nick/fast_post2.py)
import multilabel_post  # noqa: E402  (POST=multi, mo)
import input_stage as base  # noqa: E402
from models.demos.yolov4.common import get_model_result  # noqa: E402

POST = os.environ.get("POST", "raw")
POST_WORKERS = int(os.environ.get("POST_WORKERS", "2"))
DUMP_DETS = os.environ.get("DUMP_DETS")


class Submission(base.Submission):
    def setup(self, device):
        super().setup(device)
        self.post_pool = ThreadPoolExecutor(max_workers=POST_WORKERS)

    def _post(self, raw, W, H):
        if POST == "multi":
            return multilabel_post.detections_from_raw(raw, W, H, base.CONF_THRESH, base.NMS_THRESH, None)
        if POST == "raw":
            return fast_post2.detections_from_raw(raw, W, H, base.CONF_THRESH, base.NMS_THRESH, None)
        boxes, confs = get_model_result(raw, self.resolution, mesh_composer=self.out_composer)
        return fast_post.detections(boxes.numpy(), confs.numpy(), W, H, base.CONF_THRESH, base.NMS_THRESH)

    def run(self, paths):
        paths = list(paths)
        futs = {}
        chunks = [paths[i:i + base.CHUNK] for i in range(0, len(paths), base.CHUNK)]
        if not chunks:
            return {}
        pending = self._submit(chunks[0])
        for i, chunk in enumerate(chunks):
            nxt = self._submit(chunks[i + 1]) if i + 1 < len(chunks) else None
            frames = [f.result() for f in pending]
            pending = nxt
            results = self.pipeline.enqueue([t for t, _, _ in frames]).pop_all()
            for path, (_, W, H), raw in zip(chunk, frames, results):
                if POST in ("raw", "multi"):
                    futs[path] = self.post_pool.submit(self._post, raw, W, H)
                else:  # keep get_model_result on the main thread, exactly as sub_overlap does
                    boxes, confs = get_model_result(raw, self.resolution, mesh_composer=self.out_composer)
                    futs[path] = self.post_pool.submit(fast_post.detections, boxes.numpy(), confs.numpy(),
                                                       W, H, base.CONF_THRESH, base.NMS_THRESH)
        result = {p: f.result() for p, f in futs.items()}
        self.last_out = result
        return result

    def teardown(self):
        if DUMP_DETS and hasattr(self, "last_out"):
            with open(DUMP_DETS, "wb") as f:
                pickle.dump(self.last_out, f)
        pool = getattr(self, "post_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
        super().teardown()
