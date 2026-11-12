# Change log — team hawkeyett (YOLOv4, Wormhole n150)

All numbers: `harness/evaluate.py` on the 200 practice photos, fastest of the repeats, unless a row says otherwise.
**[M]** = measured on our card. **[E]** = estimated / not an end-to-end harness run. Rows are in the order we did them.
Each row is also a point on `curve.svg` where a result file exists (`curve_points/SOURCES.tsv`).

| # | Change (who) | ms/photo | mAP@50 | What it was worth | |
|---|---|---|---|---|---|
| 0 | **Start**: plain reference / fast reference (trace + 2 CQ) | 85.29 / **17.83** | .464 | starting point (`BASELINE.md`); `run_bench.sh` 320: 75.8 ms/frame, model 67.5 ms | M |
| 1 | Fast exact box step `fast_post2` (Nick) | 11.93 sequential | .464 | box step 7.30 → 4.05 ms, detections identical 200/200 | M |
| 2 | Output stage overlapped with next card pass, 2 post workers (Nick) | 8.88 | .464 | −8.9 ms vs start, identical detections | M |
| 3 | Input: decode threads build the bf16 tensor, `TORCH_THREADS=1` (Chanyoung) + merge with #2 | **5.91** | .464 | input alone 17.6 → 12.9; merged 3.0× faster than start | M |
| 4 | Accuracy cause found: port used HardMish (72 sites); real Mish restored (Mo, Nick) | 18.79 (serial) | **.5145** | +.05 mAP for ~+1 ms on the serial path (CPU fp32: .558 Mish vs .479 HardMish) | M |
| 5 | fp32 accumulation in convs + multi-label NMS (Mo, Chanyoung) | 6.56 (r320) | .5596 | +.024 for +0.22 ms; multi-label +.022 at same speed | M |
| 6 | Raw heads: ~229-op box tail off the card, Numba box step, GC freeze (Chanyoung) | 3.74 | .4656 (nb) | 5.95 → 3.74 ms; slow-run spread 2.4 → 0.07 ms | M |
| 7 | Fill/drain card passes 8,16,…,16,8 (Chanyoung) | 3.74 | .5494 | −41 ms per run (practice) | M |
| 8 | Resolution folders + 640 fix: neck conv9 read a wrong shard layout above 320 (Mo) | — | — | 640 graph correct (bottom-half PCC 0.055 → fixed); 416 unusable (13×13 grid) | M |
| 9 | **Mosaic**: 4 photos at 320 in one 640 pass (Mo, Chanyoung) | 4.05 → **2.77** | .5341 → .5581 | with raw heads + 28-config matrix; first conv bf16/HiFi4/fp32acc +.001–.006 in 13/14 pairs | M |
| 10 | **Custom Tensix kernels** (Nick): SFPU Mish, fused Mish+residual add, SPP, upsample+concat (NoC), I/O + config patch | 3.93 → 3.31 (320) | .5494 → .5545 | m640 2.87 → 2.19; −24…−31 % on all 6 configs (COCO1k); 0 fallbacks | M |
| 11 | Box-step v2: score re-rank, box voting, mosaic tile-edge penalty (Chanyoung) | 2.19 → 2.11 | .5563 → .5586 | +.0037 mean COCO1k over 6 configs (CI [+.0003, +.0065]); ~0 ms | M |
| 12 | **Final file, jury conditions** (clean env, 3 repeats) | **2.137** | **.5586** | 8.2× faster and +.095 mAP vs the reference re-measured the same day (17.59 / .464) | M |

**Did not work / rejected** (all measured unless marked): letterbox resize −7 pts (model trained on stretch) · flip TTA
+0.4 pts for 2× card work · bf16 weights + HiFi4 .445 and HiFi4 without fp32 acc .5099 (precision was not the cause) ·
on-card argmax 5.4–12.6 ms · 704 mosaic canvas (L1 overflow) · m384 mosaic (4×192) below the reference accuracy ·
stock fused conv-Mish +0.25 ms slower · Mish inside the stock conv kernel (unit tests pass, full model hangs) ·
`CV2_THREADS=1` slower · YOLOv8s / YOLO11s / YOLO11m: below the YOLOv4 frontier at equal speed (CPU fp32 mAP [M],
card device time lower bounds only [E]; YOLO11m ~10.5 ms is a branch claim [E]).

**Estimated, not measured**: YOLOv4-320 ≈ 10 GMAC (<1 ms of pure maths) [E]; the expected hour-30 score (5/5) [E];
alternative-model card times [E]. Hidden-like 800-photo draws (1.70–1.71 ms, .632–.656) are measured [M] but are our own
draws from COCO, not the official set.
