"""Drop-in replacement for post_processing() + _detections() of the fast reference.

Gives the same detections as the reference (same boxes, scores and order) on the
practice outputs, checked by test_fast_post.py. Changes are speed only:
  - row max/argmax on the int16 view of the float16 scores (same order for values >= 0)
  - loop over the classes that are present, not all 80
  - NMS from one float16 overlap matrix per class (torch), built from the same element-wise
    float16 operations as nms_cpu(); a class bigger than MATRIX_MAX uses the original loop
  - detections built from arrays, not lists of numpy scalars
"""
import numpy as np
import torch

import submission_api as api

COCO = np.asarray(api.COCO_IDS)
MATRIX_MAX = 4096  # one class with more candidates than this falls back to the loop (memory)


def _nms_loop(x1, y1, x2, y2, conf, nms_thresh):
    """nms_cpu() from post_processing.py, unchanged apart from taking columns."""
    areas = (x2 - x1) * (y2 - y1)
    order = conf.argsort()[::-1]
    keep = []
    while order.size > 0:
        i, rest = order[0], order[1:]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        over = inter / (areas[i] + areas[rest] - inter)
        order = order[np.where(over <= nms_thresh)[0] + 1]
    return np.asarray(keep, dtype=np.int64)


def _nms_matrix(x1, y1, x2, y2, conf, nms_thresh):
    """Same decisions as _nms_loop: every pair's overlap uses the identical float16 ops
    (torch, like numpy, computes each float16 op in float32 and rounds the result back)."""
    order = conf.argsort()[::-1].copy()
    t = torch.from_numpy(np.stack([x1, y1, x2, y2], axis=1)[order])
    x1, y1, x2, y2 = t.unbind(1)
    areas = (x2 - x1) * (y2 - y1)
    xx1 = torch.maximum(x1[:, None], x1[None, :])
    yy1 = torch.maximum(y1[:, None], y1[None, :])
    xx2 = torch.minimum(x2[:, None], x2[None, :])
    yy2 = torch.minimum(y2[:, None], y2[None, :])
    w = (xx2 - xx1).clamp_min(0.0)
    h = (yy2 - yy1).clamp_min(0.0)
    inter = w * h
    over = inter / (areas[:, None] + areas[None, :] - inter)
    # numpy compares a float16 array with a Python float in float16
    suppress = (~(over <= torch.tensor(nms_thresh, dtype=torch.float16))).numpy()
    # walk the kept boxes only: the next kept one is the first still alive
    alive = np.ones(len(order), dtype=bool)
    keep = []
    i = 0
    while True:
        keep.append(i)
        alive[i] = False
        alive &= ~suppress[i]
        i = int(alive.argmax())
        if not alive[i]:
            break
    return order[keep]


def detections(boxes, confs, W, H, conf_thresh, nms_thresh):
    """boxes float16 [1,N,1,4], confs float16 [1,N,80] (numpy, from get_model_result).

    Returns the harness detection dicts, identical to the reference _detections().
    """
    b = boxes[0, :, 0, :]
    c = confs[0]
    # float16 values >= 0 sort like their int16 bit patterns, so this is max/argmax without
    # converting 6300x80 values to float32. Negative or NaN maxima can't pass the threshold
    # in the reference either; rows holding a NaN are dropped below, as np.max would.
    ci = c.view(np.int16)
    row_max_bits = ci.max(axis=1)
    max_conf = row_max_bits.view(np.float16).astype(np.float32)
    sel = np.nonzero(max_conf > conf_thresh)[0]
    if sel.size == 0:
        return []
    cs = c[sel]
    nan_rows = np.isnan(cs).any(axis=1)
    if nan_rows.any():
        sel, cs = sel[~nan_rows], cs[~nan_rows]
    max_id = cs.view(np.int16).argmax(axis=1)
    l_conf = max_conf[sel]
    l_box = b[sel]

    out_box, out_conf, out_cls = [], [], []
    for j in np.unique(max_id):
        m = max_id == j
        bb = l_box[m]
        cc = l_conf[m]
        cols = (bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3])
        nms = _nms_matrix if len(cc) <= MATRIX_MAX else _nms_loop
        keep = nms(*cols, cc, nms_thresh)
        out_box.append(bb[keep])
        out_conf.append(cc[keep])
        out_cls.append(np.full(len(keep), j))

    bx = np.concatenate(out_box).astype(np.float64)
    cf = np.concatenate(out_conf).astype(np.float64)
    cl = np.concatenate(out_cls)
    score = cf * cf  # the reference stores conf twice and the harness multiplies them
    px1 = np.minimum(np.maximum(bx[:, 0] * W, 0.0), W)
    py1 = np.minimum(np.maximum(bx[:, 1] * H, 0.0), H)
    px2 = np.minimum(np.maximum(bx[:, 2] * W, 0.0), W)
    py2 = np.minimum(np.maximum(bx[:, 3] * H, 0.0), H)
    ok = ~((px2 <= px1) | (py2 <= py1))
    px1, py1, px2, py2 = px1[ok], py1[ok], px2[ok], py2[ok]
    cats = COCO[cl[ok]].tolist()
    rows = zip(cats, px1.tolist(), py1.tolist(), (px2 - px1).tolist(), (py2 - py1).tolist(), score[ok].tolist())
    return [{"category_id": k, "bbox": [x, y, w, h], "score": s} for k, x, y, w, h, s in rows]
