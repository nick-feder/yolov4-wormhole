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


# =====================================================================================================================
# Box-step variants (Chanyoung, 11 Nov 2026; ~/work/chanyoung/boxstep/BOXSTEP.md). Everything above this line is an
# unchanged copy of ~/work/chanyoung/rawheads/rawheads_nb.py. _kernel2 with default arguments returns exactly the same
# detections as _kernel (checked by the Phase 0 gate). New knobs:
#   a_obj, b_cls  rank score r = sigmoid(obj)^a * sigmoid(cls)^b   (V1; the conf > conf_t test stays on obj*cls)
#   cap           at most `cap` candidates per class enter NMS, best r first (V2; 0 = no cap)
#   method        0 hard NMS (torchvision rule), 1 linear Soft-NMS, 2 Gaussian Soft-NMS, 3 DIoU-NMS (V3)
#   sig           Gaussian Soft-NMS sigma: s *= exp(-IoU^2 / sig)
#   beta          DIoU-NMS: suppress if IoU - (d^2 / c^2)^beta > nms_t
#   floor         Soft-NMS: drop a candidate once its decayed score < floor
#   vote_t        box voting (V4): kept box = r-weighted mean of all class candidates with IoU > vote_t (0 = off)
#   vote_mode     0 = full voting (re-scan of every class candidate per kept box, ~2x kernel time)
#                 1 = fused voting (hard NMS only): mean of the kept box and the boxes IT suppresses with
#                     IoU > vote_t, accumulated inside the NMS pass (no extra loop)
#   edge_g        mosaic tile-edge penalty (V7): r *= (box area fraction inside the tile on interior sides)^edge_g
#   edge_mask     interior sides of this tile: 1 left, 2 top, 4 right, 8 bottom (0 = single photo, no penalty)
# Reported score = (final rank score)^2, the same convention as _kernel (only the order within a class matters).
# =====================================================================================================================
@nb.njit(nogil=True, cache=True, error_model="numpy")
def _kernel2(u, cell, anc, conf_t, nms_t, multi, top_k, W, H,
             a_obj, b_cls, cap, method, sig, beta, floor, vote_t, edge_g, edge_mask, vote_mode):
    ncell = u.shape[0]
    bits = np.empty(1, np.uint32)
    fv = bits.view(np.float32)
    lt = math.log(conf_t / (1.0 - conf_t))
    nrow = ncell * 3
    ccap = nrow * 80 if multi else nrow
    rb = np.empty((nrow, 4), np.float32)
    c_row = np.empty(ccap, np.int32)
    c_cls = np.empty(ccap, np.int32)
    c_s = np.empty(ccap, np.float32)
    c_so = np.empty(ccap, np.float32)
    c_sc = np.empty(ccap, np.float32)
    nr = 0
    nc = 0
    for c in range(ncell):
        for a in range(3):
            base = a * 85
            bits[0] = np.uint32(u[c, base + 4]) << 16
            obj = fv[0]
            if not (obj > lt):
                continue
            so = np.float32(1.0 / (1.0 + math.exp(-obj)))
            first = nc
            if multi:
                q = conf_t / so
                if q >= 1.0:
                    continue
                tl = math.log(q / (1.0 - q))
                for k in range(80):
                    bits[0] = np.uint32(u[c, base + 5 + k]) << 16
                    l = fv[0]
                    if l > tl - 1e-6:
                        sc = np.float32(1.0 / (1.0 + math.exp(-l)))
                        s = so * sc
                        if s > conf_t:
                            c_row[nc] = nr
                            c_cls[nc] = k
                            c_s[nc] = s
                            c_so[nc] = so
                            c_sc[nc] = sc
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
                sc = np.float32(1.0 / (1.0 + math.exp(-best)))
                s = so * sc
                if s > conf_t:
                    c_row[nc] = nr
                    c_cls[nc] = bk
                    c_s[nc] = s
                    c_so[nc] = so
                    c_sc[nc] = sc
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
    # rank score r (float64). Default a = b = 1, no edge: r = float64(conf), identical to _kernel.
    plain = a_obj == 1.0 and b_cls == 1.0
    r = np.empty(nc, np.float64)
    for i in range(nc):
        if plain:
            ri = np.float64(c_s[i])
        else:
            ri = np.float64(c_so[i]) ** a_obj * np.float64(c_sc[i]) ** b_cls
        if edge_g > 0.0 and edge_mask != 0:
            rr = c_row[i]
            x1 = np.float64(rb[rr, 0]); y1 = np.float64(rb[rr, 1])
            x2 = np.float64(rb[rr, 2]); y2 = np.float64(rb[rr, 3])
            area = (x2 - x1) * (y2 - y1)
            xl = max(x1, 0.0) if (edge_mask & 1) else x1
            yt = max(y1, 0.0) if (edge_mask & 2) else y1
            xr = min(x2, 1.0) if (edge_mask & 4) else x2
            yb = min(y2, 1.0) if (edge_mask & 8) else y2
            frac = 0.0
            if area > 0.0 and xr > xl and yb > yt:
                frac = min(1.0, (xr - xl) * (yb - yt) / area)
            ri *= frac ** edge_g
        r[i] = ri
    # sort: class ascending, rank score descending (r in [0, 1])
    key = np.empty(nc, np.float64)
    for i in range(nc):
        key[i] = c_cls[i] * 2.0 - r[i]
    order = np.argsort(key, kind="mergesort")
    sup = np.zeros(nc, np.bool_)
    sw = np.empty(nc, np.float64)          # Soft-NMS working scores
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
        e2 = e if cap <= 0 else min(e, i + cap)
        nvalid = 0
        if method == 1 or method == 2:
            for p in range(i, e2):
                sw[p] = r[order[p]]
                sup[p] = sw[p] < floor
        p = i - 1
        while True:
            # next kept candidate
            if method == 1 or method == 2:
                p = -1
                bs = -1.0
                for q in range(i, e2):
                    if not sup[q] and sw[q] > bs:
                        bs = sw[q]
                        p = q
                if p < 0:
                    break
                sup[p] = True
                ps = sw[p]
            else:
                p += 1
                while p < e2 and sup[p]:
                    p += 1
                if p >= e2:
                    break
                ps = r[order[p]]
            rp = c_row[order[p]]
            px1, py1, px2, py2 = rb[rp, 0], rb[rp, 1], rb[rp, 2], rb[rp, 3]
            ap = (px2 - px1) * (py2 - py1)
            fuse = vote_t > 0.0 and vote_mode == 1 and method == 0
            fw = ps; fx1 = ps * px1; fy1 = ps * py1; fx2 = ps * px2; fy2 = ps * py2
            for q in range(i if (method == 1 or method == 2) else p + 1, e2):
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
                iou = inter / (ap + aq - inter)
                if method == 0:
                    if iou > nms_t:
                        sup[q] = True
                    if fuse and iou > vote_t:
                        wq = r[order[q]]
                        fw += wq; fx1 += wq * qx1; fy1 += wq * qy1; fx2 += wq * qx2; fy2 += wq * qy2
                elif method == 3:
                    cw = max(px2, qx2) - min(px1, qx1)
                    ch = max(py2, qy2) - min(py1, qy1)
                    dx = (px1 + px2 - qx1 - qx2) * 0.5
                    dy = (py1 + py2 - qy1 - qy2) * 0.5
                    pen = ((dx * dx + dy * dy) / (cw * cw + ch * ch)) ** beta
                    if iou - pen > nms_t:
                        sup[q] = True
                else:
                    if method == 1:
                        if iou > nms_t:
                            sw[q] *= 1.0 - iou
                    else:
                        sw[q] *= math.exp(-iou * iou / sig)
                    if sw[q] < floor:
                        sup[q] = True
            bx1 = np.float64(px1); by1 = np.float64(py1); bx2 = np.float64(px2); by2 = np.float64(py2)
            if fuse:
                bx1 = fx1 / fw; by1 = fy1 / fw; bx2 = fx2 / fw; by2 = fy2 / fw
            elif vote_t > 0.0:
                wsum = 0.0; vx1 = 0.0; vy1 = 0.0; vx2 = 0.0; vy2 = 0.0
                for q in range(i, e2):
                    rq = c_row[order[q]]
                    qx1, qy1, qx2, qy2 = rb[rq, 0], rb[rq, 1], rb[rq, 2], rb[rq, 3]
                    iw = min(px2, qx2) - max(px1, qx1)
                    ih = min(py2, qy2) - max(py1, qy1)
                    if iw <= 0.0 or ih <= 0.0:
                        continue
                    inter = iw * ih
                    aq = (qx2 - qx1) * (qy2 - qy1)
                    if q == p or inter / (ap + aq - inter) > vote_t:
                        wq = r[order[q]]
                        wsum += wq
                        vx1 += wq * qx1; vy1 += wq * qy1; vx2 += wq * qx2; vy2 += wq * qy2
                if wsum > 0.0:
                    bx1 = vx1 / wsum; by1 = vy1 / wsum; bx2 = vx2 / wsum; by2 = vy2 / wsum
            X1 = min(max(bx1 * W, 0.0), W)
            Y1 = min(max(by1 * H, 0.0), H)
            X2 = min(max(bx2 * W, 0.0), W)
            Y2 = min(max(by2 * H, 0.0), H)
            if X2 > X1 and Y2 > Y1:
                nvalid += 1
                o_cls[no] = cls
                o_box[no, 0] = X1
                o_box[no, 1] = Y1
                o_box[no, 2] = X2 - X1
                o_box[no, 3] = Y2 - Y1
                o_s[no] = ps * ps
                no += 1
                if top_k > 0 and nvalid >= top_k:
                    break
        i = e
    return o_cls[:no], o_box[:no], o_s[:no]


V2_DEFAULTS = dict(a=1.0, b=1.0, cap=0, m=0, sig=0.5, beta=1.0, floor=0.001, vote=0.0, edge=0.0, vm=0)


def detections_arrays2(u, W, H, conf_thresh, nms_thresh, multi=False, top_k=100, edge_mask=0, **kw):
    """_kernel2 wrapper; kw keys = V2_DEFAULTS (a, b, cap, m, sig, beta, floor, vote, edge, vm)."""
    p = {**V2_DEFAULTS, **kw}
    cell, anc = _tables(u.shape[0])
    return _kernel2(u, cell, anc, float(conf_thresh), float(nms_thresh), bool(multi), int(top_k or 0),
                    float(W), float(H), float(p["a"]), float(p["b"]), int(p["cap"]), int(p["m"]),
                    float(p["sig"]), float(p["beta"]), float(p["floor"]), float(p["vote"]),
                    float(p["edge"]), int(edge_mask), int(p["vm"]))


def warm2(**kw):
    """compile (or load the cache) of _kernel2 outside the timed region."""
    u = np.zeros((2100, 256), np.uint16)
    for m in (False, True):
        detections_arrays2(u, 640, 480, 0.001, 0.45, m, 100, 0, **kw)
