"""nick_k: Nick's custom Tensix kernels (ttnn.generic_op) on the 320 / 384 / 512 / 640 graphs (combo_k).
Kernels: kernels/compute_mish2.cpp (Mish V14 bfp8 / V34 = V14 + RNE bf16 rounding for bf16), kernels/compute_mishadd.cpp
(fused ResBlock tail), reader_push*.cpp (zero-copy L1 shard CBs). Neck kernels from combo_neck:
patch_upcat.py + upcat.py (kernels/upcat_dm_v3.cpp bfp8 @320, upcat_g.cpp bf16 @384+), patch_spp.py + spp.py
(kernels/*_spp.cpp @320, *_spp_g.cpp bf16 @384+); bit-exact data movement / max.
"""