"""HawkeyeTT submission for one input size. Self-contained: this folder's own model copy (shadow/),
a snapshot of the team pipeline (input_stage, fast_post2, team_submission), and the accuracy fixes.
Change RES only by copying the folder; tune anything else freely inside this folder."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "shadow"))      # this folder's model copy wins over ~/tt-metal/models/demos/yolov4
sys.path.insert(0, HERE)
sys.path.insert(1, "/home/user/challenge/harness")
RES = 640

import ttnn
import models.tt_cnn.tt.builder as builder
# Accuracy fix 1: real Mish (the model was trained with it; the port used HardMish). +5 mAP at 320.
builder._POST["HARDMISH"] = lambda x, p: ttnn.mish(x, fast_and_approximate_mode=True)
# Accuracy fix 2: fp32 accumulation in every conv. +2.4 mAP at 320, and makes 640 work.
def _compute_config(cfg, device):
    return ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=cfg.math_fidelity,
        fp32_dest_acc_en=True, packer_l1_acc=cfg.packer_l1_acc)
builder.to_compute_config = _compute_config

import models.demos.yolov4.common as common
assert HERE in common.__file__, f"wrong model copy loaded: {common.__file__}"
import input_stage, fast_post2
from box_rows import box_rows
input_stage.RESOLUTION = (RES, RES)
fast_post2._BOX_ROWS[3 * ((RES // 8) ** 2 + (RES // 16) ** 2 + (RES // 32) ** 2)] = box_rows(RES)
from team_pipeline import Submission  # noqa: E402,F401
