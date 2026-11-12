"""fast_post.detections() with a cheaper NMS. Same interface.

Changes against fast_post.py:
  - one pass over all present classes (no per-class torch calls): candidates are ordered
    class by class, each class in the reference order (conf.argsort()[::-1]), and pairs from
    different classes are masked out
  - blocked greedy NMS: each block of BLOCK candidates is compared with the boxes already
    kept (only ~18% survive) and with itself, instead of building the full N x N matrix.
    Every overlap uses the same float16 operations as nms_cpu(), so the decisions are the same.
  - top_k (optional): stop a class once top_k valid boxes are kept. cocomap scores only the
    top 100 per (image, class), so top_k=100 changes the list but not the score.
"""
import numpy as np
import torch

import submission_api as api

COCO = np.asarray(api.COCO_IDS)
BLOCK = 256


def _suppress(r, c, ar, ac, thr):
    """r [R,4], c [C,4] float16 boxes; True where IoU(r_i, c_j) > thr, computed like nms_cpu()."""
    xx1 = torch.maximum(r[:, 0, None], c[None, :, 0])
    yy1 = torch.maximum(r[:, 1, None], c[None, :, 1])
    xx2 = torch.minimum(r[:, 2, None], c[None, :, 2])
    yy2 = torch.minimum(r[:, 3, None], c[None, :, 3])
    w = (xx2 - xx1).clamp_min(0.0)
    h = (yy2 - yy1).clamp_min(0.0)
    inter = w * h
    over = inter / (ar[:, None] + ac[None, :] - inter)
    return ~(over <= thr)


def detections(boxes, confs, W, H, conf_thresh, nms_thresh, top_k=None):
    b = boxes[0, :, 0, :]
    c = confs[0]
    ci = c.view(np.int16)
    max_conf = ci.max(axis=1).view(np.float16).astype(np.float32)
    sel = np.nonzero(max_conf > conf_thresh)[0]
    if sel.size == 0:
        return []
    return _from_selected(sel, c[sel], max_conf, b[sel], W, H, nms_thresh, top_k)


# get_model_result() unpads the boxes by taking these rows, keyed by the number of score rows
_BOX_ROW = np.r_[0:6100, 6128:6228, 6256:6356]                              # 320x320
_BOX_ROWS = {6300: _BOX_ROW, 25200: np.r_[0:24400, 24428:24828, 24856:25256]}  # 640x640


def detections_from_raw(raw, W, H, conf_thresh, nms_thresh, top_k=None):
    """Same output as get_model_result() + detections(), straight from the pipeline's host tensors.

    The card returns both outputs as row-major bfloat16. get_model_result() converts all 6300x80
    scores to float16; here only the selected rows are converted. That gives the same values
    because bf16 -> fp16 is monotonic (so max commutes with it) and is applied elementwise.
    """
    c16 = raw[1].to_torch()[0].view(torch.int16).numpy()   # bf16 bit patterns [6300, 80] (25200 at 640)
    bits = c16.max(axis=1)        # non-negative bf16 sort like their int16 bits, like fp16
    max_conf = torch.from_numpy(bits).view(torch.bfloat16).to(torch.float16).numpy().astype(np.float32)
    sel = np.nonzero(max_conf > conf_thresh)[0]
    if sel.size == 0:
        return []
    cs = torch.from_numpy(c16[sel]).view(torch.bfloat16).to(torch.float16).numpy()
    bx = raw[0].to_torch()[0, 0]                             # [6384, 4] bf16, padded
    l_box = bx[torch.from_numpy(_BOX_ROWS[len(c16)][sel])].to(torch.float16).numpy()
    return _from_selected(sel, cs, max_conf, l_box, W, H, nms_thresh, top_k)


def _from_selected(sel, cs, max_conf, l_box, W, H, nms_thresh, top_k):
    nan_rows = np.isnan(cs).any(axis=1)
    if nan_rows.any():
        sel, cs, l_box = sel[~nan_rows], cs[~nan_rows], l_box[~nan_rows]
        if sel.size == 0:
            return []
    max_id = cs.view(np.int16).argmax(axis=1)
    l_conf = max_conf[sel]

    # candidate order: class ascending, inside a class exactly the reference's argsort()[::-1]
    by_cls = np.argsort(max_id, kind="stable")
    cls_sorted = max_id[by_cls]
    starts = np.flatnonzero(np.r_[True, cls_sorted[1:] != cls_sorted[:-1]])
    ends = np.r_[starts[1:], len(by_cls)]
    order = np.concatenate([by_cls[s:e][l_conf[by_cls[s:e]].argsort()[::-1]] for s, e in zip(starts, ends)])
    cls = max_id[order]
    conf = l_conf[order]
    box16 = np.ascontiguousarray(l_box[order])
    n = len(order)
    class_end = np.repeat(ends, ends - starts)  # for each candidate, where its class ends

    bx = box16.astype(np.float64)
    px1 = np.minimum(np.maximum(bx[:, 0] * W, 0.0), W)
    py1 = np.minimum(np.maximum(bx[:, 1] * H, 0.0), H)
    px2 = np.minimum(np.maximum(bx[:, 2] * W, 0.0), W)
    py2 = np.minimum(np.maximum(bx[:, 3] * H, 0.0), H)
    valid = ~((px2 <= px1) | (py2 <= py1))

    t = torch.from_numpy(box16)
    area = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    tcls = torch.from_numpy(cls)
    thr = torch.tensor(nms_thresh, dtype=torch.float16)

    keep = []           # positions in candidate order, ascending
    n_valid = {}        # class -> valid kept so far (only used with top_k)
    pos = 0
    while pos < n:
        end = min(n, pos + BLOCK)
        blk = slice(pos, end)
        alive = np.ones(end - pos, dtype=bool)
        same = (tcls[blk, None] == tcls[None, blk])
        S = (_suppress(t[blk], t[blk], area[blk], area[blk], thr) & same).numpy()
        if keep:
            # kept boxes from classes before this block's first class can't matter
            kp = np.asarray(keep)
            kp = kp[cls[kp] >= cls[pos]]
            if kp.size:
                kt = torch.from_numpy(kp)
                Sk = _suppress(t[kt], t[blk], area[kt], area[blk], thr) & (tcls[kt, None] == tcls[None, blk])
                alive &= ~Sk.any(0).numpy()
        jump = end
        while True:
            i = int(alive.argmax())
            if not alive[i]:
                break
            p = pos + i
            keep.append(p)
            alive[i] = False
            alive &= ~S[i]
            if top_k is not None and valid[p]:
                k = cls[p]
                n_valid[k] = n_valid.get(k, 0) + 1
                if n_valid[k] >= top_k:
                    alive[(cls[blk] == k)] = False
                    jump = max(jump, class_end[p])
        pos = jump

    keep = np.asarray(keep, dtype=np.int64)
    keep = keep[valid[keep]]
    cf = conf[keep].astype(np.float64)
    score = cf * cf  # the reference stores conf twice and the harness multiplies them
    x1, y1, x2, y2 = px1[keep], py1[keep], px2[keep], py2[keep]
    cats = COCO[cls[keep]].tolist()
    rows = zip(cats, x1.tolist(), y1.tolist(), (x2 - x1).tolist(), (y2 - y1).tolist(), score.tolist())
    return [{"category_id": k, "bbox": [x, y, w, h], "score": s} for k, x, y, w, h, s in rows]
