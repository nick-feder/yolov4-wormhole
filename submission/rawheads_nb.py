"""GIL-free host box step: raw heads -> decode -> per-class NMS -> top-k, in ONE numba call (Chanyoung).

Why: with the card at ~3.3 ms/frame the torch box step (rawheads_decode.decode + fast_post2.detections_tv)
is ~30-40 small torch calls per frame. Every call drops and re-takes the GIL, and under contention
with the main/decoder threads each re-take can wait for the switch interval: measured 3x slowdowns
and single frames stalling for ~0.45 s (RAW_HEADS.md 5.8-5.9). Here the whole step is one
@njit(nogil=True) call; only the final list of dicts is built in Python.

Semantics (same as rawheads_decode.decode + fast_post2.detections_tv):
  * decode = CPU reference YoloLayer in fp32 (scale_x_y = 1), boxes normalised xyxy
  * conf = sigmoid(obj) * sigmoid(cls); single-label: best class only; multi-label: every class > t
  * NMS per class, greedy in score order, suppress if IoU > nms_thresh (torchvision rule)
  * boxes clipped to the image, empty ones dropped, at most top_k kept per class (cocomap scores
    only the top 100 per (image, class)), score reported as conf*conf (reference convention)
The bf16 head values are read bit-exactly (uint16 << 16), no full fp32 conversion of the tensor.
"""
import math

import numba as nb
import numpy as np
import torch

_ANCHORS = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
_STRIDES = [8, 16, 32]
_TABS = {}


def _tables(ncell):
    """per cell: cx, cy, 1/G; per (cell, anchor): anchor w/G, h/G. Grids from the cell count (2100 -> 40,20,10)."""
    if ncell not in _TABS:
        g0 = int(round((ncell / 1.3125) ** 0.5))
        cell, anc = [], []
        for s, G in enumerate((g0, g0 // 2, g0 // 4)):
            l = np.arange(G * G)
            cell.append(np.stack([l % G, l // G, np.full(G * G, 1.0 / G)], 1))
            a = np.array([[_ANCHORS[(3 * s + k) * 2] / _STRIDES[s] / G, _ANCHORS[(3 * s + k) * 2 + 1] / _STRIDES[s] / G]
                          for k in range(3)])
            anc.append(np.broadcast_to(a, (G * G, 3, 2)))
        _TABS[ncell] = (np.ascontiguousarray(np.concatenate(cell), np.float64),
                        np.ascontiguousarray(np.concatenate(anc), np.float64))
    return _TABS[ncell]


@nb.njit(nogil=True, cache=True, error_model="numpy")
def _kernel(u, cell, anc, conf_t, nms_t, multi, top_k, W, H):
    ncell = u.shape[0]
    bits = np.empty(1, np.uint32)
    fv = bits.view(np.float32)
    lt = math.log(conf_t / (1.0 - conf_t))
    nrow = ncell * 3
    cap = nrow * 80 if multi else nrow
    rb = np.empty((nrow, 4), np.float32)
    c_row = np.empty(cap, np.int32)
    c_cls = np.empty(cap, np.int32)
    c_s = np.empty(cap, np.float32)
    nr = 0
    nc = 0
    for c in range(ncell):
        for a in range(3):
            base = a * 85
            bits[0] = np.uint32(u[c, base + 4]) << 16
            obj = fv[0]
            if not (obj > lt):            # sigmoid(obj) > t is necessary for any class to pass
                continue
            so = np.float32(1.0 / (1.0 + math.exp(-obj)))
            first = nc
            if multi:
                q = conf_t / so
                if q >= 1.0:
                    continue
                tl = math.log(q / (1.0 - q))  # sigmoid(l) > q  <=>  l > logit(q); exact test below
                for k in range(80):
                    bits[0] = np.uint32(u[c, base + 5 + k]) << 16
                    l = fv[0]
                    if l > tl - 1e-6:
                        s = so * np.float32(1.0 / (1.0 + math.exp(-l)))
                        if s > conf_t:
                            c_row[nc] = nr
                            c_cls[nc] = k
                            c_s[nc] = s
                            nc += 1
            else:
                best = -np.inf
                bk = 0
                for k in range(80):
                    bits[0] = np.uint32(u[c, base + 5 + k]) << 16
                    l = fv[0]
                    if l > best:
                        best = l
                        bk = k
                s = so * np.float32(1.0 / (1.0 + math.exp(-best)))
                if s > conf_t:
                    c_row[nc] = nr
                    c_cls[nc] = bk
                    c_s[nc] = s
                    nc += 1
            if nc == first:
                continue
            bits[0] = np.uint32(u[c, base + 0]) << 16
            tx = fv[0]
            bits[0] = np.uint32(u[c, base + 1]) << 16
            ty = fv[0]
            bits[0] = np.uint32(u[c, base + 2]) << 16
            tw = fv[0]
            bits[0] = np.uint32(u[c, base + 3]) << 16
            th = fv[0]
            x = (1.0 / (1.0 + math.exp(-tx)) + cell[c, 0]) * cell[c, 2]
            y = (1.0 / (1.0 + math.exp(-ty)) + cell[c, 1]) * cell[c, 2]
            w = math.exp(tw) * anc[c, a, 0]
            h = math.exp(th) * anc[c, a, 1]
            x1 = x - w * 0.5
            y1 = y - h * 0.5
            rb[nr, 0] = x1
            rb[nr, 1] = y1
            rb[nr, 2] = x1 + w
            rb[nr, 3] = y1 + h
            nr += 1
    # sort candidates: class ascending, score descending (scores are in (0, 1))
    key = np.empty(nc, np.float64)
    for i in range(nc):
        key[i] = c_cls[i] * 2.0 - c_s[i]
    order = np.argsort(key, kind="mergesort")
    sup = np.zeros(nc, np.bool_)
    o_cls = np.empty(nc, np.int32)
    o_box = np.empty((nc, 4), np.float64)
    o_s = np.empty(nc, np.float64)
    no = 0
    i = 0
    while i < nc:
        cls = c_cls[order[i]]
        e = i
        while e < nc and c_cls[order[e]] == cls:
            e += 1
        nvalid = 0
        for p in range(i, e):
            if sup[p]:
                continue
            rp = c_row[order[p]]
            px1, py1, px2, py2 = rb[rp, 0], rb[rp, 1], rb[rp, 2], rb[rp, 3]
            ap = (px2 - px1) * (py2 - py1)
            for q in range(p + 1, e):
                if sup[q]:
                    continue
                rq = c_row[order[q]]
                qx1, qy1, qx2, qy2 = rb[rq, 0], rb[rq, 1], rb[rq, 2], rb[rq, 3]
                iw = min(px2, qx2) - max(px1, qx1)
                ih = min(py2, qy2) - max(py1, qy1)
                if iw <= 0.0 or ih <= 0.0:
                    continue
                inter = iw * ih
                aq = (qx2 - qx1) * (qy2 - qy1)
                if inter / (ap + aq - inter) > nms_t:
                    sup[q] = True
            X1 = min(max(np.float64(px1) * W, 0.0), W)
            Y1 = min(max(np.float64(py1) * H, 0.0), H)
            X2 = min(max(np.float64(px2) * W, 0.0), W)
            Y2 = min(max(np.float64(py2) * H, 0.0), H)
            if X2 > X1 and Y2 > Y1:
                nvalid += 1
                cf = np.float64(c_s[order[p]])
                o_cls[no] = cls
                o_box[no, 0] = X1
                o_box[no, 1] = Y1
                o_box[no, 2] = X2 - X1
                o_box[no, 3] = Y2 - Y1
                o_s[no] = cf * cf
                no += 1
                if top_k > 0 and nvalid >= top_k:
                    break
        i = e
    return o_cls[:no], o_box[:no], o_s[:no]


_COCO = None


def _coco():
    global _COCO
    if _COCO is None:
        import submission_api as api
        _COCO = np.asarray(api.COCO_IDS)
    return _COCO


def heads_u16(raw):
    """pipeline output (rmcat host tensor, list of 3, or torch) -> [Ncell, 256] uint16 bf16 bit patterns."""
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    ts = [r.to_torch() if hasattr(r, "to_torch") else r for r in raw]
    ts = [t.reshape(-1, t.shape[-1]).to(torch.bfloat16) for t in ts]
    t = ts[0] if len(ts) == 1 else torch.cat(ts, 0)
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def detections_arrays(u, W, H, conf_thresh, nms_thresh, multi=False, top_k=100):
    cell, anc = _tables(u.shape[0])
    return _kernel(u, cell, anc, float(conf_thresh), float(nms_thresh), bool(multi), int(top_k or 0),
                   float(W), float(H))


def detections(raw, W, H, conf_thresh, nms_thresh, multi=False, top_k=100):
    cls, box, s = detections_arrays(heads_u16(raw), W, H, conf_thresh, nms_thresh, multi, top_k)
    cats = _coco()[cls].tolist()
    b = box.tolist()
    return [{"category_id": k, "bbox": bb, "score": sc} for k, bb, sc in zip(cats, b, s.tolist())]


def warm():
    """compile (or load the cache) outside the timed region."""
    u = np.zeros((2100, 256), np.uint16)
    for m in (False, True):
        detections_arrays(u, 640, 480, 0.001, 0.45, m, 100)
