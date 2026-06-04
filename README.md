# HyperCP - Hyperedge-Aware Conformal Prediction for Supply-Chain Forecasting

Anonymized code release accompanying the paper submission. It contains the
model, calibration methods, baselines, evaluation metrics, the
SupplyGraph-Stressed benchmark, and scripts that reproduce every results
table in the paper.

> Anonymized for double-blind review: no author, affiliation, or funding
> information is included.

## Contents
- `hypercp/` - core package
  - `models/` - DeepSets+HGNN+TCN encoder, forecast & quantile heads, joint
    model, hyperedge (worst-case-firm) conformity score
  - `calibration/` - split-CP, node/hyperedge ACI, Conformal-PID, AgACI,
    CF-GNN, NCPNET
  - `data/` - SupplyGraph and M5 loaders, Forrester-Sterman simulator,
    SupplyGraph-Stressed generator
  - `eval/` - metrics (WMAPE, CRPS, PICP, PINAW, ECE, Spearman)
  - `baselines/` - TFT, LightGBM-Quantile, CCF-HGNN
- `scripts/` - experiment runners (gates 1-3, baselines, LGBM wrapper)
- `make_all.py` - regenerate every results JSON
- `artifacts/supplygraph_stressed.npz` - the released SupplyGraph-Stressed benchmark
- `requirements.txt`, `Dockerfile` - environment

## Install
```
pip install -r requirements.txt      # numpy, pandas, scipy, torch, lightgbm, matplotlib
# or:  docker build -t hypercp .     # Python 3.10 + dependencies
```
Pure PyTorch - no PyTorch Geometric or DGL dependency.

## Data
- **SupplyGraph** (public; Wasi et al., arXiv 2401.15299): place the dataset at
  `SupplyGraph-main/Raw Dataset/Homogenoeus/` (the layout the loader expects;
  see `hypercp/data/supplygraph.py`).
- **M5** (public; Kaggle "M5 Forecasting - Accuracy"): place
  `sales_train_evaluation.csv` under `m5-forecasting/`.
- **SupplyGraph-Stressed** (this work): shipped as
  `artifacts/supplygraph_stressed.npz`, and regenerable from the deterministic
  generator in `hypercp/data/supplygraph_stressed.py`.

## Reproduce the paper

| Result | Command |
|---|---|
| Coverage on SupplyGraph (5 seeds) | `python scripts/run_gate1.py --dataset supplygraph --n_seeds 5 --train_epochs 30 --output runs/gate1_supplygraph.json` |
| M5 generalization (3 seeds) | `python scripts/run_gate1.py --dataset m5 --n_seeds 3 --m5_subsample_n 1000 --m5_subsample_days 500 --output runs/gate1_m5.json` |
| Forecasting baselines | `python scripts/run_baselines.py --n_seeds 3 --baselines tft lgbm ccf --output runs/baselines.json` |
| HyperCP on LightGBM-Q wrapper | `python scripts/run_hypercp_on_lgbm.py --dataset supplygraph --n_seeds 3 --output runs/hypercp_on_lgbm.json` |
| Resilience grounding | `python scripts/run_gate2.py --n_shocks 200 --output runs/gate2.json` |
| Sample efficiency | `python scripts/run_gate3.py --n_seeds 3 --output runs/gate3.json` |
| Everything | `python make_all.py`  (add `--quick` for a CPU smoke test) |

Add `--device cuda:0` for GPU. Each seed of the joint forecaster trains in
~22 s on an NVIDIA L4; the full suite runs in under an hour.

## Reproducibility notes
- The conformal methods and the LightGBM-Q base forecaster are exactly
  reproducible across runs. The TFT baseline uses non-deterministic cuDNN
  kernels and may vary by ~+/-0.01-0.02 WMAPE; no reported claim depends on
  its point estimate.
- Hyperparameters are fixed (no search): Adam lr 1e-3, weight decay 5e-4,
  batch 16, 30 epochs, hidden-dim 64, ACI eta=0.05, calibration window 30,
  quantile levels {.05,.1,.5,.9,.95}, target alpha=0.10.
- The full proof of Theorem 1 and the reproducibility checklist are in the
  paper's supplementary material.
