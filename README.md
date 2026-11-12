# YOLOv4 on Tenstorrent Wormhole n150: team HawkeyeTT

Our changes for the SEMICON / TUM Venture Labs hackathon (Tenstorrent track): faster YOLOv4 inference
with mAP@50 preserved.

**Hand-in configuration "m640":** four 320×320 photos per 640×640 pass, custom kernels K1–K4, raw
heads decoded on the host (box step v2), multi-label NMS 0.45, conf 0.001.
Result on the 800-photo jury set: **~1.71 ms / photo, mAP@50 0.656**.

## Layout

| Path | What |
|---|---|
| `submission/submission_final.py` | Hand-in entry point (md5 `85eb92da4f437ff752a648c602da6f4b`) |
| `submission/submission_k.py` | Installs the custom kernels on top of `submission.py` |
| `submission/submission.py` | Pinned configuration (RES=640, MOSAIC=1, ACC=0, PREC_FIRST=1) |
| `submission/nick_k/` | Custom Tensix kernels (C++) and their Python host code |
| `submission/shadow/` | Patched copy of the tt-metal YOLOv4 model code (Apache-2.0, Tenstorrent) |
| `docs/handin/` | Change log, settings, trade-off curve and measured points |
| `docs/explainers/` | Technical deep dive and animated chip dataflow (open in a browser) |

## Custom kernels

| Kernel | What it does |
|---|---|
| K1 | Custom SFPU Mish activation |
| K2 | Fused ResBlock tail: residual + Mish(conv), intermediate stays in registers |
| K3 | SPP as one multicore op (cascaded 5×5 pools give 9×9 and 13×13; bit-exact) |
| K4 | Upsample ×2 + concat as pure NoC data movement into the next conv's layout (byte-exact) |

## Run

Requires tt-metal at commit `5a3ee49ca03` and the challenge harness. Not included: the model weights
(`yolov4.pth`, ~258 MB) and sample images. Put them back at the same paths under `submission/shadow/`
(they come from the stock tt-metal YOLOv4 demo).

```bash
bash ~/challenge/jury_hour30.sh <team> submission/submission_final.py
```
