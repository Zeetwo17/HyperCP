"""
HyperCP joint-model training driver.

Production training entry point for the multi-task joint model (encoder +
forecast head + quantile head, FAMO/Kendall/equal weighting). Supports:

- Three datasets: SupplyGraph (default), M5, SCR/SupplySim
- Mixed-precision training (torch.cuda.amp) — automatic on GPU, no-op on CPU
- Checkpointing on best validation pinball loss
- Resume from checkpoint
- Optional W&B logging
- Rolling-origin time-series sampling for training

Usage
-----
# Train on SupplyGraph with default config (CPU OK for small smoke run):
$ python scripts/train.py --dataset supplygraph --epochs 50

# Full M5 training on GPU with mixed precision:
$ python scripts/train.py \\
      --dataset m5 \\
      --device cuda:0 \\
      --epochs 50 \\
      --batch_size 32 \\
      --amp \\
      --checkpoint_dir runs/m5_baseline \\
      --wandb_project hypercp-icdm2026

# Resume from checkpoint:
$ python scripts/train.py \\
      --dataset supplygraph \\
      --resume runs/sg_baseline/checkpoint_best.pt

# CPU smoke test (no GPU required):
$ python scripts/train.py --dataset supplygraph --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Make the repo root importable so `hypercp` resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch import nn

from hypercp.data.supplygraph import SupplyGraphHypergraph
from hypercp.data.m5 import M5Hypergraph
from hypercp.models import (
    EncoderConfig,
    ForecastHeadConfig,
    QuantileHeadConfig,
    JointConfig,
    HyperCPJointModel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Run config + checkpoint format.
# =============================================================================


@dataclass
class TrainConfig:
    """Top-level training configuration."""
    dataset: str = "supplygraph"           # 'supplygraph' or 'm5'
    device: str = "auto"                   # 'auto', 'cuda:0', 'cpu', ...
    epochs: int = 50
    batch_size: int = 16                   # # of rolling-origin windows per minibatch
    lr: float = 1e-3
    weight_decay: float = 5e-4
    hidden_dim: int = 64
    n_hgnn_layers: int = 4
    weighting: str = "famo"                # 'equal', 'kendall', 'famo'
    horizon: int = 4
    window: int = 5                        # T = rolling-origin window length
    amp: bool = False                      # mixed-precision (cuda only)
    grad_clip: float = 1.0                 # max-norm grad clip
    seed: int = 0
    checkpoint_dir: Optional[str] = None
    resume: Optional[str] = None
    save_every: int = 10                   # save checkpoint every N epochs
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    log_every: int = 1                     # log every N epochs
    # Dataset-specific.
    m5_subsample_n: Optional[int] = None
    m5_subsample_days: Optional[int] = None


# =============================================================================
# Data loaders.
# =============================================================================


def load_dataset(cfg: TrainConfig):
    """Return (model_input_features, incidence, partition_vec, splits, target_channel)."""
    if cfg.dataset == "supplygraph":
        sg = SupplyGraphHypergraph()
        return {
            "features": sg.node_features,           # (n_firms, T, F)
            "incidence": sg.hyperedge_incidence_matrix(),
            "partition": sg.hyperedge_partition_vector(),
            "splits": sg.rolling_origin_split(),
            "n_firms": sg.n_products,
            "n_features": sg.n_channels,
            "n_timesteps": sg.n_timesteps,
            "target_channel": 1,                    # sales_order in SupplyGraph
            "raw": sg,
        }
    elif cfg.dataset == "m5":
        m5 = M5Hypergraph(
            subsample_n=cfg.m5_subsample_n,
            subsample_days=cfg.m5_subsample_days,
        )
        return {
            "features": m5.node_features,
            "incidence": m5.hyperedge_incidence_matrix(),
            "partition": m5.hyperedge_partition_vector(),
            "splits": m5.rolling_origin_split(),
            "n_firms": m5.n_skus,
            "n_features": m5.n_channels,
            "n_timesteps": m5.n_timesteps,
            "target_channel": 0,                    # M5 has only one channel
            "raw": m5,
        }
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")


# =============================================================================
# Window batching.
# =============================================================================


def sample_windows(
    features: torch.Tensor,
    train_range: range,
    window: int,
    horizon: int,
    batch_size: int,
    target_channel: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample `batch_size` rolling-origin windows from the train range.

    Returns
    -------
    features_batch : (batch_size, n_firms, window, n_features)
    targets        : (batch_size, n_firms, horizon)  -- diagonal "own product"
    mask           : (batch_size, n_firms, n_firms, horizon)  -- identity-on-products
    """
    n_firms = features.size(0)
    n_features = features.size(2)
    valid_starts = list(range(train_range.start, train_range.stop - window - horizon))
    if not valid_starts:
        raise ValueError(
            f"Train range too short for window={window}, horizon={horizon}: "
            f"need at least {window + horizon} timesteps in train."
        )

    starts = torch.randint(
        low=0, high=len(valid_starts), size=(batch_size,), generator=rng
    )
    feat_batch = []
    tgt_batch = []
    for i in starts:
        s = valid_starts[int(i)]
        feat = features[:, s:s + window, :]                         # (n_firms, T, F)
        tgt = features[:, s + window:s + window + horizon, target_channel]  # (n_firms, H)
        feat_batch.append(feat)
        tgt_batch.append(tgt)
    feat_batch_t = torch.stack(feat_batch, dim=0)                   # (B, n_firms, T, F)
    tgt_batch_t = torch.stack(tgt_batch, dim=0)                     # (B, n_firms, H)
    # Diagonal target -- each firm forecasts only its own product.
    tgt_full = tgt_batch_t.unsqueeze(2).expand(
        batch_size, n_firms, n_firms, horizon
    )  # broadcast across product axis
    eye = torch.eye(n_firms, device=features.device).unsqueeze(-1)
    mask = eye.unsqueeze(0).expand(batch_size, n_firms, n_firms, horizon)
    return feat_batch_t, tgt_full, mask


# =============================================================================
# Train / eval loops.
# =============================================================================


@torch.no_grad()
def evaluate(
    model: HyperCPJointModel,
    features: torch.Tensor,
    eval_range: range,
    window: int,
    horizon: int,
    target_channel: int,
    device: torch.device,
) -> dict:
    """Compute WMAPE and pinball loss on the evaluation range."""
    model.eval()
    n_firms = features.size(0)

    total_huber = 0.0
    total_pinball = 0.0
    n_windows = 0
    abs_errs = []
    abs_targets = []

    valid_starts = list(range(eval_range.start, eval_range.stop - window - horizon))
    for s in valid_starts:
        feat = features[:, s:s + window, :].unsqueeze(0).to(device)
        tgt = features[:, s + window:s + window + horizon, target_channel]  # (n_firms, H)
        tgt_full = tgt.unsqueeze(1).expand(n_firms, n_firms, horizon).unsqueeze(0).to(device)
        eye = torch.eye(n_firms, device=device).unsqueeze(-1).unsqueeze(0)
        mask = eye.expand(1, n_firms, n_firms, horizon)

        outputs = model(feat)
        losses = model.compute_losses(outputs, tgt_full, mask)
        total_huber += float(losses["huber"])
        total_pinball += float(losses["pinball"])
        n_windows += 1

        # WMAPE numerator/denominator.
        idx = torch.arange(n_firms, device=device)
        y_pred_diag = outputs["y_hat"][0, idx, idx, :]  # (n_firms, H)
        abs_errs.append((y_pred_diag - tgt.to(device)).abs().sum().item())
        abs_targets.append(tgt.abs().sum().item())

    avg_huber = total_huber / max(n_windows, 1)
    avg_pinball = total_pinball / max(n_windows, 1)
    wmape = sum(abs_errs) / max(sum(abs_targets), 1e-8)
    return {
        "huber": avg_huber,
        "pinball": avg_pinball,
        "wmape": wmape,
        "n_windows": n_windows,
    }


def train_one_run(cfg: TrainConfig) -> dict:
    """Single training run (no seed-loop). Returns final-metrics dict."""
    # ----- 1. Device and seed -----
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )
    torch.manual_seed(cfg.seed)
    logger.info("Device: %s", device)
    logger.info("Mixed precision (AMP): %s",
                cfg.amp and device.type == "cuda")

    # ----- 2. Data -----
    data = load_dataset(cfg)
    n_firms = data["n_firms"]
    n_features = data["n_features"]
    train_range, val_range, test_range = data["splits"]
    incidence = data["incidence"].to(device)
    features = data["features"].to(device)
    logger.info(
        "Dataset=%s: n_firms=%d, T=%d, n_features=%d, "
        "train=[%d,%d) val=[%d,%d) test=[%d,%d)",
        cfg.dataset, n_firms, data["n_timesteps"], n_features,
        train_range.start, train_range.stop,
        val_range.start, val_range.stop,
        test_range.start, test_range.stop,
    )

    # ----- 3. Model -----
    enc_cfg = EncoderConfig(
        n_features=n_features, hidden_dim=cfg.hidden_dim,
        n_hgnn_layers=cfg.n_hgnn_layers,
    )
    fhd_cfg = ForecastHeadConfig(
        hidden_dim=cfg.hidden_dim, n_products=n_firms, horizon=cfg.horizon,
    )
    qhd_cfg = QuantileHeadConfig(
        hidden_dim=cfg.hidden_dim, n_products=n_firms, horizon=cfg.horizon,
    )
    model = HyperCPJointModel(JointConfig(
        encoder=enc_cfg, forecast=fhd_cfg, quantile=qhd_cfg,
        weighting=cfg.weighting,
    )).to(device)
    model.set_hypergraph(incidence)
    logger.info("Model params: %d", model.n_parameters())

    # ----- 4. Optimiser & AMP scaler -----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ----- 5. Resume from checkpoint -----
    start_epoch = 0
    best_val_pinball = float("inf")
    if cfg.resume is not None:
        ckpt_path = Path(cfg.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_pinball = ckpt.get("best_val_pinball", float("inf"))
        logger.info(
            "Resumed from %s at epoch %d (best_val_pinball=%.4f).",
            ckpt_path, start_epoch, best_val_pinball,
        )

    # ----- 6. W&B (optional) -----
    wandb_run = None
    if cfg.wandb_project is not None:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                config=asdict(cfg),
            )
        except ImportError:
            logger.warning(
                "wandb not installed; install with `pip install wandb` or remove --wandb_project."
            )
            wandb_run = None

    # ----- 7. Training loop -----
    ckpt_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Iterations per "epoch": cover the training window once on average.
    n_train_windows = max(
        1, len(train_range) - cfg.window - cfg.horizon
    )
    steps_per_epoch = max(1, n_train_windows // cfg.batch_size)
    rng = torch.Generator(device="cpu").manual_seed(cfg.seed)

    history = []
    t_start = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_huber = 0.0
        epoch_pinball = 0.0
        for step in range(steps_per_epoch):
            feat_b, tgt_b, mask_b = sample_windows(
                features.cpu(), train_range, cfg.window, cfg.horizon,
                cfg.batch_size, data["target_channel"], rng,
            )
            feat_b = feat_b.to(device)
            tgt_b = tgt_b.to(device)
            mask_b = mask_b.to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(feat_b)
                    losses = model.compute_losses(outputs, tgt_b, mask_b)
                    total, weights = model.weighter.combine(losses)
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(feat_b)
                losses = model.compute_losses(outputs, tgt_b, mask_b)
                total, weights = model.weighter.combine(losses)
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            model.weighter.update(losses)

            epoch_huber += float(losses["huber"].detach())
            epoch_pinball += float(losses["pinball"].detach())

        epoch_huber /= steps_per_epoch
        epoch_pinball /= steps_per_epoch

        # ----- Validate -----
        val_metrics = evaluate(
            model, features, val_range, cfg.window, cfg.horizon,
            data["target_channel"], device,
        )

        if val_metrics["pinball"] < best_val_pinball:
            best_val_pinball = val_metrics["pinball"]
            if ckpt_dir is not None:
                ckpt_path = ckpt_dir / "checkpoint_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_pinball": best_val_pinball,
                    "config": asdict(cfg),
                }, ckpt_path)
                logger.info("Saved best checkpoint to %s", ckpt_path)

        # Periodic snapshot.
        if ckpt_dir is not None and (epoch + 1) % cfg.save_every == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_pinball": best_val_pinball,
                "config": asdict(cfg),
            }, ckpt_path)

        record = {
            "epoch": epoch,
            "train_huber": epoch_huber,
            "train_pinball": epoch_pinball,
            "val_huber": val_metrics["huber"],
            "val_pinball": val_metrics["pinball"],
            "val_wmape": val_metrics["wmape"],
            "best_val_pinball": best_val_pinball,
            "weights": weights,
            "elapsed": time.time() - t_start,
        }
        history.append(record)

        if (epoch + 1) % cfg.log_every == 0:
            logger.info(
                "epoch %3d/%d | train h=%.4f p=%.4f | val h=%.4f p=%.4f wmape=%.3f | best=%.4f | %.1fs",
                epoch + 1, cfg.epochs,
                epoch_huber, epoch_pinball,
                val_metrics["huber"], val_metrics["pinball"], val_metrics["wmape"],
                best_val_pinball, record["elapsed"],
            )

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train/huber": epoch_huber,
                "train/pinball": epoch_pinball,
                "val/huber": val_metrics["huber"],
                "val/pinball": val_metrics["pinball"],
                "val/wmape": val_metrics["wmape"],
                "best_val_pinball": best_val_pinball,
                **{f"weight/{k}": v for k, v in weights.items()},
            })

    # ----- 8. Final test eval -----
    if ckpt_dir is not None and (ckpt_dir / "checkpoint_best.pt").exists():
        ckpt = torch.load(
            ckpt_dir / "checkpoint_best.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(ckpt["model"])

    test_metrics = evaluate(
        model, features, test_range, cfg.window, cfg.horizon,
        data["target_channel"], device,
    )
    logger.info(
        "TEST: huber=%.4f pinball=%.4f wmape=%.3f",
        test_metrics["huber"], test_metrics["pinball"], test_metrics["wmape"],
    )

    if wandb_run is not None:
        wandb_run.log({
            "test/huber": test_metrics["huber"],
            "test/pinball": test_metrics["pinball"],
            "test/wmape": test_metrics["wmape"],
        })
        wandb_run.finish()

    return {
        "test": test_metrics,
        "history": history,
        "best_val_pinball": best_val_pinball,
    }


# =============================================================================
# CLI.
# =============================================================================


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="HyperCP training driver.")
    parser.add_argument("--dataset", choices=["supplygraph", "m5"], default="supplygraph")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_hgnn_layers", type=int, default=4)
    parser.add_argument("--weighting", choices=["equal", "kendall", "famo"], default="famo")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--m5_subsample_n", type=int, default=None)
    parser.add_argument("--m5_subsample_days", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="CPU-friendly tiny config; overrides epoch/batch/dim.")
    args = parser.parse_args()

    cfg = TrainConfig(**{
        k: v for k, v in vars(args).items() if k != "smoke"
    })
    if args.smoke:
        cfg.epochs = 5
        cfg.batch_size = 4
        cfg.hidden_dim = 32
        cfg.amp = False
        if cfg.dataset == "m5":
            cfg.m5_subsample_n = 200
            cfg.m5_subsample_days = 100
    return cfg


def main() -> None:
    cfg = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger.info("Config: %s", cfg)
    result = train_one_run(cfg)

    # Save run history.
    if cfg.checkpoint_dir is not None:
        out = Path(cfg.checkpoint_dir) / "history.json"
        out.write_text(json.dumps({
            "test": result["test"],
            "best_val_pinball": result["best_val_pinball"],
            "history": result["history"],
            "config": asdict(cfg),
        }, indent=2, default=str))
        logger.info("Saved history to %s", out)


if __name__ == "__main__":
    main()
