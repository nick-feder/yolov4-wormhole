# Hand-in — team hawkeyett, Tenstorrent track (YOLOv4 on Wormhole n150)

| Item asked for | Where |
|---|---|
| 1. Submission file (the one configuration we claim) | `~/work/chanyoung/final_m640/submission_final.py` |
| 2. One-page change log, measured vs estimated | `CHANGELOG.md` (this folder) |
| 2b. Trade-off curve (`harness/plot_curve.py`) | `curve.svg` (all points), `curve_zoom.svg` (without the 85 ms plain reference) |
| 2c. Result files behind every point | `curve_points/*.json`, list with original paths in `curve_points/SOURCES.tsv` |
| 3. Settings stated | section "Settings" below |
| 4. Our own starting measurement, before any change | section "Starting measurement" below, raw files in `~/winners_doc/MAIN_README/baseline/` |

## 1. Submission file

```
bash ~/challenge/jury_hour30.sh hawkeyett ~/work/chanyoung/final_m640/submission_final.py
```

- md5 `85eb92da4f437ff752a648c602da6f4b`. Check the folder with `cd ~/work/chanyoung/final_m640 && md5sum -c MD5SUMS.txt`.
- It is one entry file built on the harness `Submission` API. It loads its helper modules (raw-head model, box step,
  custom kernels in `nick_k/`) from its **own folder**, so the folder must stay together. Run it by its path; do not
  copy the single file somewhere else.
- No environment variables needed: kernels and box-step v2 are the file defaults.
- Tested exactly like the jury script (clean environment, `PYTHONPATH=~/tt-metal`, 20-photo smoke + 3 timed repeats):
  practice **2.137 ms/photo, mAP@50 .5586**; kernel counters 0 fallbacks; 66 stress sets incl. 280 broken/empty files,
  0 crashes. Details: `~/work/chanyoung/final_test/FINAL_M640.md`.

## 3. Settings

| Setting | Value |
|---|---|
| Model | **YOLOv4**, the supplied tt-metal weights, unchanged. Activation restored to real Mish (the port used HardMish). |
| Resolution | **640 × 640 model input = 2 × 2 mosaic of four photos, each resized (stretched) to 320 × 320**. One card pass per 4 photos. Each photo therefore has 320-pixel detail. |
| Confidence threshold | **0.001** (same floor as the reference; low on purpose, the AP walk needs the low-score boxes) |
| Overlap (NMS IoU) threshold | **0.45**, hard NMS, multi-label (every class above the confidence floor per box), at most 100 boxes kept per class |
| Box score | `sigmoid(obj)^0.5 · sigmoid(cls)^1.25`, box voting for overlaps IoU > 0.75, score × 0.5 near the internal mosaic edges |
| Numerics | weights bfp8, activations bf16, LoFi matrix math; first conv bf16 + HiFi4 + fp32 accumulation |
| Card / host | trace + 2 command queues, 4 custom Tensix kernels, raw heads decoded on the host (Numba), 8 decode workers |

What they do: a bigger input gives more detail per object (more mAP) but costs card time; the mosaic keeps 320 detail
per photo while sharing one pass over 4 photos. A lower confidence floor adds low-score boxes, which can only add AP but
costs box-step time. A lower NMS IoU removes more overlapping boxes (fewer duplicates, but more merged neighbours).

## 4. Starting measurement (10 Nov, before anyone changed anything)

| Run | ms/photo | mAP@50 | Tool |
|---|---|---|---|
| Plain reference `submission_reference.py` | 85.29 | .464 | `harness/evaluate.py`, practice, fastest of 3 |
| Fast reference `submission_reference_fast.py` | **17.83** | **.464** | same |
| Per-stage bench, plain path, one frame at a time | 75.81 at 320 (decode 2.40, prepare 0.74, upload 3.15, model 67.53, boxes 2.00); 88.86 at 640 | — | `bench/run_bench.sh` |

Where the 17.5 ms of the fast reference went: NMS 8.6 ms (49 %), card 5.0 (29 %), `from_torch` 2.1 (12 %),
unpad 0.9 (5 %). Board n150 L, firmware 19.11.0.0, tt-metal 5a3ee49ca03, Python 3.10.19.
Same-day re-measure of the fast reference under jury conditions: 17.59 ms / .464.

## Result

| | Reference (same day) | Ours, final file | Change |
|---|---|---|---|
| Practice (200) | 17.59 ms / .464 | **2.137 ms / .5586** | **8.2× faster, +.095 mAP** |

Speed: `harness/evaluate.py` timing (the jury's timer) for every end-to-end number; `run_bench.sh` for the per-stage
starting breakdown. Accuracy: `harness/evaluate.py` scoring on the practice photos. Which numbers are estimated is marked
in `CHANGELOG.md`.
