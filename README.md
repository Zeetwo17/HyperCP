# HyperCP — Hyperedge-Aware Conformal Prediction for Supply-Chain Forecasting

**Authors:** [Vishal Pandey](https://github.com/Zeetwo17) · [Sahil Kumar](https://github.com/sahil1418)
[![Paper](https://img.shields.io/badge/ICDM%202026-Accepted-blue)](https://icdm2026.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)

Official code release for the paper accepted at **IEEE ICDM 2026** (Research Track).

HyperCP is a conformal-prediction wrapper that lifts Adaptive Conformal
Inference (ACI) from individual nodes to **hyperedges** in a supply-chain
hypergraph. By taking the worst-case (max) residual within each hyperedge, it
provides a **finite-sample marginal coverage guarantee** at the hyperedge
granularity — the first such guarantee for supply-chain forecasting.

## Key Results

| Dataset | Method | PICP (↑ 0.90) | PINAW (↓) |
|---------|--------|:---:|:---:|
| SupplyGraph | Split CP (node) | 0.873 ± 0.004 | 0.149 ± 0.005 |
| SupplyGraph | **ACI (hyperedge, ours)** | **0.962 ± 0.004** | **0.156 ± 0.002** |
| SupplyGraph | + 1/√\|e\| weight | 0.956 ± 0.006 | 0.152 ± 0.004 |
| M5 (1000 SKUs) | **ACI (hyperedge, ours)** | **0.989** | **0.407** |
| M5 (1000 SKUs) | + 1/√\|e\| weight | 0.947 | 0.201 |

## Repository Structure

```
hypercp/
├── models/          # DeepSets+HGNN+TCN encoder, forecast & quantile heads,
│                    # joint model, hyperedge conformity score (max aggregation)
├── calibration/     # Split-CP, ACI (node/hyperedge), Conformal-PID, AgACI,
│                    # CF-GNN-style, NCPNET-style
├── data/            # SupplyGraph and M5 loaders, Forrester-Sterman simulator,
│                    # SupplyGraph-Stressed generator
├── eval/            # Metrics (PICP, PINAW, ECE, CRPS), block bootstrap CIs,
│                    # simultaneous hyperedge coverage
└── baselines/       # TFT, LightGBM-Quantile, CCF-HGNN

scripts/             # Gate 1/2/3 experiment runners, LightGBM wrapper
tests/               # ACI direction unit tests
make_all.py          # Regenerate every results JSON
artifacts/           # SupplyGraph-Stressed benchmark (.npz)
```

## Installation

```bash
pip install -r requirements.txt      # numpy, pandas, scipy, torch, lightgbm, matplotlib

# Or use Docker for a reproducible environment:
docker build -t hypercp .
docker run --gpus all hypercp python make_all.py
```

Pure PyTorch — **no** PyTorch Geometric or DGL dependency.

## Data Setup

| Dataset | Source | Path |
|---------|--------|------|
| **SupplyGraph** | [Wasi et al. (arXiv 2401.15299)](https://arxiv.org/abs/2401.15299) | `SupplyGraph-main/Raw Dataset/Homogenoeus/` |
| **M5** | [Kaggle M5 Forecasting](https://www.kaggle.com/c/m5-forecasting-accuracy) | `m5-forecasting/sales_train_evaluation.csv` |
| **SupplyGraph-Stressed** | This work (shipped) | `artifacts/supplygraph_stressed.npz` |

## Reproduce the Paper

| Result | Command |
|--------|---------|
| Gate 1: Coverage on SupplyGraph (5 seeds) | `python scripts/run_gate1.py --dataset supplygraph --n_seeds 5 --train_epochs 30 --output runs/gate1_supplygraph.json` |
| Gate 1: M5 generalization (3 seeds) | `python scripts/run_gate1.py --dataset m5 --n_seeds 3 --m5_subsample_n 1000 --m5_subsample_days 500 --output runs/gate1_m5.json` |
| Forecasting baselines | `python scripts/run_baselines.py --n_seeds 3 --baselines tft lgbm ccf --output runs/baselines.json` |
| HyperCP on LightGBM-Q wrapper | `python scripts/run_hypercp_on_lgbm.py --dataset supplygraph --n_seeds 3 --output runs/hypercp_on_lgbm.json` |
| Gate 2: Resilience grounding | `python scripts/run_gate2.py --n_shocks 200 --output runs/gate2.json` |
| Gate 3: Sample efficiency | `python scripts/run_gate3.py --n_seeds 3 --output runs/gate3.json` |
| **Everything** | `python make_all.py` (add `--quick` for CPU smoke test) |

Add `--device cuda:0` for GPU. Each seed trains in ~22s on an NVIDIA L4; the full suite runs in under an hour.

## Hyperparameters

All hyperparameters are fixed (no search):

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 5e-4 |
| Batch size | 16 |
| Epochs | 30 |
| Hidden dim | 64 |
| ACI η | 0.05 |
| Calibration window | 30 |
| Quantile levels | {0.05, 0.10, 0.50, 0.90, 0.95} |
| Target α | 0.10 |

## Tests

```bash
python -m pytest tests/ -v
```

Verifies the ACI update direction is correct (Gibbs–Candès convention:
`α_{t+1} = α_t + η(α_target − err_t)`).

## Reproducibility Notes

- Conformal methods and LightGBM-Q are exactly reproducible across runs.
- TFT uses non-deterministic cuDNN kernels and may vary by ~±0.01 WMAPE;
  no reported claim depends on its point estimate.
- The full proof of Theorem 1 and reproducibility checklist are in the
  paper's supplementary material.

## Citation

```bibtex
@inproceedings{hypercp2026,
  title     = {HyperCP: Hyperedge-Aware Conformal Prediction for
               Supply-Chain Forecasting},
  booktitle = {IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```

## License

This project is licensed under the MIT License.
