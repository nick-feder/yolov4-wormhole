"""Hand-in candidate (Chanyoung, 11 Nov 2026): m640_a0p1 + Nick's Tensix kernels + host box step v2.

Same code as ~/work/combos/m640_a0p1/submission_k.py (this folder is a byte-identical copy, see MD5SUMS.txt).
The only difference: BOXV=v2 is the default here, because jury_hour30.sh sets no environment variables.
An explicit BOXV in the environment still wins. Kernels are on by default in submission_k.py.
"""
import importlib.util
import os
import sys

os.environ.setdefault("BOXV", "v2")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location("final_submission_k", os.path.join(HERE, "submission_k.py"))
_k = importlib.util.module_from_spec(_spec)
sys.modules["final_submission_k"] = _k
_spec.loader.exec_module(_k)
Submission = _k.Submission
