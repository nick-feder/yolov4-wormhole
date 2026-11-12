"""This folder's submission.py + Nick's custom Tensix kernels (nick_k/, combo_k). Drop-in for evaluate.py.

Loads the folder's own `submission.py` (same dir, unchanged: pinned RES / MOSAIC / ACC / PREC_FIRST, raw heads,
patch_cfg H18, library Mish fast-mode hook), then applies, each switchable via env (default 1 = on):

  kmish2   KM2 [30]       custom SFPU Mish on every conv-post Mish (V14 on bfp8, KM2_BF16 [34] = V14+RNE on bf16)
  mishadd  MISHADD [1]    fused ResBlock tail res + mish(conv) (23 per forward); FB_MA_V [30] bfp8 / FB_MA_BF16 [34] bf16 math
  upcat    FB_UPCAT [1]   nick_k/patch_upcat.py (combo_neck upcat / upcat_g), only if that file exists
  spp      FB_SPP [1]     nick_k/patch_spp.py (combo_neck spp / spp_g), only if that file exists

Order kmish2 -> mishadd -> upcat -> spp (as final_best/sub_best.py): patch_upcat.apply() itself re-points its module-level
`ttnn` to patch_cfg's neck `_Proxy` if present; spp then proxies the globals of the current TtNeck.__call__ (= upcat's),
so it must come after upcat and nothing may overwrite patch_upcat.ttnn afterwards. Unsupported tensors fall back to the
stock op (counters at exit: every `fallback` should be 0). NK_LOG=1 prints every Mish / ResBlock-tail site once.
"""
import atexit
import importlib
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_base = _load("combo_base_submission", os.path.join(HERE, "submission.py"))
_on = lambda k, d="1": os.environ.get(k, d).strip() not in ("0", "")
_report = {"cfg": getattr(_base, "CFG_STATS", None)}

from nick_k import patch_kmish2, patch_mishadd  # noqa: E402

_report["kmish2"] = patch_kmish2.apply()      # KM2=0 -> no-op
if _on("MISHADD"):
    _report["mishadd"] = patch_mishadd.apply()

_NK = os.path.join(HERE, "nick_k")
if _on("FB_UPCAT") and os.path.exists(os.path.join(_NK, "patch_upcat.py")):
    patch_upcat = importlib.import_module("nick_k.patch_upcat")
    patch_upcat.apply()  # re-points patch_upcat.ttnn to patch_cfg's neck _Proxy itself
    _report["upcat"] = patch_upcat.STATS

if _on("FB_SPP") and os.path.exists(os.path.join(_NK, "patch_spp.py")):
    patch_spp = importlib.import_module("nick_k.patch_spp")
    patch_spp.apply()
    _report["spp"] = patch_spp.STATS

print(f"[submission_k] {os.path.basename(HERE)} components: {[k for k, v in _report.items() if v is not None]} "
      f"KM2={os.environ.get('KM2', '30')} KM2_BF16={os.environ.get('KM2_BF16', '34')} "
      f"FB_MA_V={os.environ.get('FB_MA_V', '30')} FB_MA_BF16={os.environ.get('FB_MA_BF16', '34')} "
      f"FB_UPCAT={os.environ.get('FB_UPCAT', '1')} FB_SPP={os.environ.get('FB_SPP', '1')}", flush=True)
atexit.register(lambda: print(f"[submission_k] counters (incl. compile/capture): {_report}", flush=True))

Submission = _base.Submission
