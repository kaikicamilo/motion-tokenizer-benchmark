# Vendored SnapMoGen code

Subset of https://github.com/snap-research/SnapMoGen (commit `dc6881d`, 2025-09-25),
redistributed under the Snap Inc. Non-Commercial License (see `LICENSE`).
Only the modules required by this benchmark are kept: the HRVQVAE backbone,
the shared quantizer, the tokenizer trainer, both dataset loaders, both
retrieval evaluators and their utilities. MoMask++ generation, GMR, transformer
and retargeting code was removed.

## Modified upstream files

| File | Change |
|---|---|
| `model/vq/rvq_model.py` | quantizer selection for the benchmarked methods |
| `config/load_config.py` | `_base_` config inheritance |
| `utils/get_opt.py` | dataset root from `HUMANML3D_ROOT` instead of a hard-coded path |
| `utils/motion_process_hml.py`, `common/animation.py` | NumPy 2 compatibility |
| `utils/utils.py` | Matplotlib 3.10 compatibility |
| `trainers/__init__.py` | drops imports of removed trainers |

To see the exact changes, diff this directory against upstream at that commit:

```bash
git clone https://github.com/snap-research/SnapMoGen && cd SnapMoGen && git checkout dc6881d
diff -ru . ../motion-tokenizer-benchmark/third_party/snapmogen
```

## Added files (ours)

`model/vq/fsq_quantizer.py`, `lfq_quantizer.py`, `bsq_quantizer.py`,
`residual_fsq_quantizer.py`, `rvq_independent.py`.
