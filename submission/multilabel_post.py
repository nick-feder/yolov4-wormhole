"""Multi-label box step (POST=multi): every (box, class) pair above the confidence threshold is a
candidate, not only each box's best class. Class-aware NMS via torchvision.batched_nms (C++, releases
the GIL), pre-filtered to the TOPK best pairs, at most MAX_DET kept per photo.
CPU fp32 test at 320: 0.557 -> 0.567 mAP@50. Same raw-input interface as fast_post2.detections_from_raw."""
import os
import numpy as np
import torch, torchvision
import submission_api as api
from fast_post2 import _BOX_ROWS

COCO = torch.tensor(api.COCO_IDS)
TOPK = int(os.environ.get("ML_TOPK", "3000"))
MAX_DET = int(os.environ.get("ML_MAXDET", "300"))


def detections_from_raw(raw, W, H, conf_thresh, nms_thresh, top_k=None):
    c = raw[1].to_torch()[0].float()                       # [N, 80] class scores (obj * cls)
    i, k = (c > conf_thresh).nonzero(as_tuple=True)
    if i.numel() == 0:
        return []
    s = c[i, k]
    if s.numel() > TOPK:
        s, o = s.topk(TOPK); i, k = i[o], k[o]
    bx = raw[0].to_torch()[0, 0]                           # padded [rows, 4], normalized x1 y1 x2 y2
    rows = torch.from_numpy(_BOX_ROWS[c.shape[0]])
    b = bx[rows[i]].float() * torch.tensor([W, H, W, H], dtype=torch.float32)
    b[:, 0::2] = b[:, 0::2].clamp(0, W); b[:, 1::2] = b[:, 1::2].clamp(0, H)
    ok = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    b, s, k = b[ok], s[ok], k[ok]
    keep = torchvision.ops.batched_nms(b, s, k, nms_thresh)[:MAX_DET]
    b, s, k = b[keep], s[keep], COCO[k[keep]]
    xywh = torch.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1).tolist()
    return [{"category_id": int(ci), "bbox": bb, "score": float(sc)} for ci, bb, sc in zip(k.tolist(), xywh, s.tolist())]
