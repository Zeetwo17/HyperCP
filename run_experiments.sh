#!/bin/bash
# =================================================================
# HyperCP Camera-Ready Experiment Runner — Lightning AI
# =================================================================
#
# Prerequisites (do ONCE before running this script):
#   1. Upload the hypercp_anon_release.zip and unzip it
#   2. Download SupplyGraph dataset:
#        git clone https://github.com/ciol-researchlab/SupplyGraph.git SupplyGraph-main
#      (or place it so SupplyGraph-main/Raw Dataset/Homogenoeus/ exists)
#   3. Download M5 dataset (optional, for Table III):
#        Place sales_train_evaluation.csv in m5-forecasting/
#
# Usage:
#   cd hypercp_anon_release
#   chmod +x run_experiments.sh
#   bash run_experiments.sh
#
# Estimated runtimes (NVIDIA L4 / T4):
#   Step 1 (SupplyGraph Gate 1, 5 seeds): ~5 min
#   Step 2 (M5 Gate 1, 3 seeds):          ~8 min  [SKIP if no M5 data]
#   Step 3 (HyperCP on LightGBM, 3 seeds): ~3 min
#   Step 4 (Gate 2, resilience):           ~5 min
#   Step 5 (Gate 3, sample efficiency):    ~3 min
#   TOTAL: ~25 min with GPU
# =================================================================

set -e

echo "================================================"
echo "HyperCP Camera-Ready Experiment Runner"
echo "================================================"

# Install dependencies
pip install -r requirements.txt 2>/dev/null || true

# Create output directory
mkdir -p runs

# Detect device
DEVICE="cpu"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && DEVICE="cuda:0"
echo "Using device: $DEVICE"

# ---- Step 0: Run unit tests ----
echo ""
echo "=== Step 0: Unit tests ==="
python -m pytest tests/test_aci_direction.py -v
echo "[OK] All ACI direction tests passed"

# ---- Step 1: SupplyGraph Gate 1 (Table I, 5 seeds) ----
echo ""
echo "=== Step 1: SupplyGraph Gate 1 (5 seeds) ==="
echo "Estimated: ~5 min on GPU"
time python scripts/run_gate1.py \
    --dataset supplygraph \
    --device "$DEVICE" \
    --n_seeds 5 \
    --train_epochs 30 \
    --batch_size 16 \
    --hidden_dim 64 \
    --output runs/gate1_supplygraph_fixed.json

# ---- Step 2: M5 Gate 1 (Table III, 3 seeds) ----
if [ -d "m5-forecasting" ]; then
    echo ""
    echo "=== Step 2: M5 Gate 1 (3 seeds) ==="
    echo "Estimated: ~8 min on GPU"
    time python scripts/run_gate1.py \
        --dataset m5 \
        --device "$DEVICE" \
        --n_seeds 3 \
        --train_epochs 30 \
        --batch_size 16 \
        --hidden_dim 64 \
        --m5_subsample_n 1000 \
        --m5_subsample_days 500 \
        --output runs/gate1_m5_fixed.json
else
    echo ""
    echo "=== Step 2: SKIPPED (m5-forecasting/ not found) ==="
fi

# ---- Step 3: HyperCP on LightGBM wrapper (Table V, 3 seeds) ----
echo ""
echo "=== Step 3: HyperCP on LightGBM (3 seeds) ==="
echo "Estimated: ~3 min"
time python scripts/run_hypercp_on_lgbm.py \
    --dataset supplygraph \
    --device "$DEVICE" \
    --n_seeds 3 \
    --output runs/hypercp_on_lgbm_fixed.json

# ---- Step 4: Gate 2 resilience ----
echo ""
echo "=== Step 4: Gate 2 resilience ==="
echo "Estimated: ~5 min"
time python scripts/run_gate2.py \
    --device "$DEVICE" \
    --n_shocks 200 \
    --output runs/gate2_fixed.json

# ---- Step 5: Gate 3 sample efficiency ----
echo ""
echo "=== Step 5: Gate 3 sample efficiency (3 seeds) ==="
echo "Estimated: ~3 min"
time python scripts/run_gate3.py \
    --device "$DEVICE" \
    --n_seeds 3 \
    --output runs/gate3_fixed.json

echo ""
echo "================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "================================================"
echo "Output JSONs in runs/:"
ls -la runs/*.json
echo ""
echo "Next steps:"
echo "  1. Download runs/*.json from Lightning AI"
echo "  2. Compare with old numbers"
echo "  3. Update paper tables"
