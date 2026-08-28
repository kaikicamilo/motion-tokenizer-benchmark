# Quantizing Human Motion: A Controlled Study of Trade-offs in Motion Tokenizers

Code for a controlled comparison of discrete motion tokenizers — VQ-VAE, FSQ,
BSQ, LFQ, MS-RVQ, shared-codebook RVQ, Residual FSQ and RVQ with independent
codebooks — on **SnapMoGen** and **HumanML3D**.

All methods share the same encoder/decoder (attention-augmented temporal
convolutional autoencoder, ~23.6M parameters), the same 2048-state per-stage
vocabulary and the same training recipe; only the quantizer changes.

This repository contains **code only**. Datasets, evaluator weights,
checkpoints and result tables are not included, the paper's numbers are
reproduced by running the scripts below.

## Layout

```
configs/
  snapmogen/     base.yaml + one file per method (vqvae, fsq, bsq, lfq, hrvqvae, hrvqvae_8421,
  humanml3d/     rvq, rvq4, rvq_independent, rvq_independent4, residual_fsq)
scripts/
  train.py                 train one method on one dataset (both taken from the config)
  run_all.sh               train every method of a dataset, sequentially
  collect_metrics.py       table metrics from the training logs  -> metrics_<dataset>.csv
  eval_variance.py         FID / R-Precision mean ± std over repeated evaluation rounds
  eval_mpjpe_humanml3d.py  MPJPE (cm) from the saved checkpoints (HumanML3D)
  plot_radar.py            paper figure
  benchmark.py             shared helpers (method registry, model construction)
figures/                   paper figure
third_party/snapmogen/     vendored SnapMoGen code (backbone, trainer, loaders, evaluators);
                           see its NOTICE.md for the license and the list of modifications
```

Method file names follow the original SnapMoGen configs: `hrvqvae*` is MS-RVQ,
`rvq*` is shared-codebook RVQ, `rvq_independent*` is RVQ with one codebook per stage.

## Setup

```bash
pip install -r requirements.txt
```

Everything runs from the repository root, which doubles as the runtime
workspace. Create these (all git-ignored):

```
data/SnapMoGen/       SnapMoGen release      — https://github.com/snap-research/SnapMoGen
data/HumanML3D/       rebuilt HumanML3D      — https://github.com/EricGuo5513/HumanML3D
checkpoint_dir/       evaluator weights (below) + training outputs
glove/                GloVe vocabulary used by the HumanML3D evaluator
```

Evaluator weights, in the layout produced by SnapMoGen's `prepare/download_evaluators.sh`
and `prepare/download_glove.sh`:

```
checkpoint_dir/snapmogen/evaluator/eval_klde-5_late-5_nlayer6_norm/{evaluator.yaml, model/net_best_top1.tar}
checkpoint_dir/humanml3d/Comp_v6_KLD005/{opt.txt, meta/}
checkpoint_dir/humanml3d/text_mot_match/model/finest.tar
glove/our_vab_{data.npy, idx.pkl, words.pkl}
```

Dataset roots are set in `configs/<dataset>/base.yaml` (`data.root_dir`); the
HumanML3D evaluator additionally reads `HUMANML3D_ROOT` if set.

## Reproducing the paper

```bash
# 1. train (one run per method; logs/<dataset>/<method>.log, checkpoints under checkpoint_dir/)
python scripts/train.py --config configs/snapmogen/rvq_independent.yaml
bash scripts/run_all.sh snapmogen
bash scripts/run_all.sh humanml3d

# 2. table metrics from the logs (lowest-FID checkpoint per method)
python scripts/collect_metrics.py --dataset snapmogen --logs logs/snapmogen --out metrics_snapmogen.csv
python scripts/eval_mpjpe_humanml3d.py --out mpjpe_humanml3d.csv
python scripts/collect_metrics.py --dataset humanml3d --logs logs/humanml3d --out metrics_humanml3d.csv --mpjpe mpjpe_humanml3d.csv

# 3. mean ± std of FID and R-Precision over 20 evaluation rounds (the values reported in the tables)
python scripts/eval_variance.py --dataset snapmogen --rounds 20 --out variance_snapmogen.json
python scripts/eval_variance.py --dataset humanml3d --rounds 20 --out variance_humanml3d.json

# 4. figure
python scripts/plot_radar.py --metrics metrics_snapmogen.csv
```

## Protocol notes

- Each run reports the checkpoint with the lowest reconstruction FID, evaluated
  every 10 epochs. HumanML3D is evaluated on its test split, SnapMoGen on its
  validation split; within a dataset the same split selects the checkpoint for
  every method.
- R-Precision uses the frozen text–motion retrieval model released with each
  dataset (pool of 100 for SnapMoGen, 32 for HumanML3D). A single evaluation
  pass is noisy — captions, temporal crops and pools are resampled — so the
  paper reports the mean over 20 rounds (`eval_variance.py`).
- MPJPE is in centimetres on both datasets. Perplexity is the validation-pass
  value at the reported epoch and measures codebook utilization, not quality.
- Tokens per latent timestep: the encoder downsamples time by 4×; flat residual
  methods emit one token per stage, MS-RVQ emits Σ 1/s_i (1.5 for [2,1],
  1.875 for [8,4,2,1]).
- LFQ collapses to a single code under this backbone at every setting we tried
  and is reported as a finding.

## License

Our code is released for research use. `third_party/snapmogen/` is
redistributed under the **Snap Inc. Non-Commercial License** (kept verbatim in
`third_party/snapmogen/LICENSE`), which governs the use of this repository as a
whole; modifications to upstream files are listed in
`third_party/snapmogen/NOTICE.md`.
