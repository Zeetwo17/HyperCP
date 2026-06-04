"""
HyperCP encoder: DeepSets feature encoder + HGNN convolutions + TCN.

Implements the shared backbone of Section 4.1 of the paper:
    SharedEncoder = DeepSets(features) -> [HGNN]_{L layers} -> TCN(temporal)

Aligned faithfully with SC-RIHN (Shen et al., AAAI 2026):
- DeepSets feature encoder with learnable positional embeddings per feature
  index, exactly per their Eq. (2).
- HGNN convolution: Z^(l+1) = sigma( D_v^(-1/2) H W_e D_e^(-1) H^T
  D_v^(-1/2) Z^(l) W^(l) ), exactly per their Eq. (3).
- Default L = 4 hypergraph layers (SC-RIHN's reported optimum).

Two changes from SC-RIHN, justified in the paper:
1. Multi-channel feature input (F=5 channels: production, sales_order,
   delivery, factory_issue, predicted_imbalance). SC-RIHN uses F=1.
2. Temporal block (TCN) replaces SC-RIHN's mean-pool readout over T
   timesteps. Dilations (1, 2, 4) over a 3-layer dilated-causal-conv stack
   give the encoder access to longer temporal context without inflating
   parameter count.

The module is fully device-portable: all parameters and buffers move with
`encoder.to(device)`. Smoke tests run on CPU with tiny networks; production
training uses GPU via the same code path.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


# =============================================================================
# Config dataclass.
# =============================================================================


@dataclass
class EncoderConfig:
    """Hyperparameters for the HyperCP encoder backbone.

    Defaults match SC-RIHN's reported optimum (L=4, hidden_dim=64) so the
    Phase 2 replication gate uses these exact values.
    """
    n_features: int = 5            # F: number of input channels per timestep
    hidden_dim: int = 64           # d: SC-RIHN reports 64 as optimum
    n_hgnn_layers: int = 4         # L: SC-RIHN reports 4 layers optimum
    n_tcn_layers: int = 3          # K: TCN depth (dilations 1, 2, 4)
    tcn_kernel: int = 3            # kernel size for causal convolutions
    dropout: float = 0.1           # dropout between layers
    pool: str = "mean"             # 'mean' or 'sum' for DeepSets aggregation
    feature_mlp_hidden_mult: int = 2  # phi MLP hidden = hidden_dim * mult
    use_layer_norm: bool = True    # LayerNorm after each HGNN layer
    activation: str = "gelu"       # 'gelu' or 'relu'
    temporal_readout: str = "tcn"  # 'tcn' (default, ours) or 'mean_pool' (SC-RIHN)


def _make_activation(name: str) -> nn.Module:
    return {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
    }[name.lower()]


# =============================================================================
# DeepSets feature encoder (SC-RIHN Eq. 2).
# =============================================================================


class DeepSetsFeatureEncoder(nn.Module):
    """Per-firm DeepSets-style feature encoder.

    Given a feature vector x_c ∈ R^F per firm, compute:
        h_c = Pool( { phi(x_c[d], pos(d)) | d = 1..F } )
    where pos(d) ∈ R^hidden is a learnable positional embedding per feature
    index and phi is a shared MLP.

    This mirrors SC-RIHN's Eq. 2 verbatim. The set-based pooling makes the
    encoder invariant to the order of feature dimensions, which lets us mix
    real SupplyGraph channels with synthetic-derived channels without
    committing to a specific ordering convention.

    Input shape:  (..., n_firms, F)     -- variable leading batch dims
    Output shape: (..., n_firms, hidden_dim)
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Positional embedding pos(d) for each feature index d in {0..F-1}.
        self.feature_pos_embed = nn.Embedding(cfg.n_features, cfg.hidden_dim)

        # Shared MLP phi.
        hidden = cfg.hidden_dim * cfg.feature_mlp_hidden_mult
        # Input to phi: feature scalar (1) + positional embedding (hidden_dim)
        self.phi = nn.Sequential(
            nn.Linear(1 + cfg.hidden_dim, hidden),
            _make_activation(cfg.activation),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.hidden_dim),
        )

        # Pooling.
        self.pool_kind = cfg.pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features.

        Parameters
        ----------
        x : torch.Tensor, shape (..., n_firms, F)
            Per-firm feature vectors. Leading batch dims (e.g. time) are
            preserved.

        Returns
        -------
        h : torch.Tensor, shape (..., n_firms, hidden_dim)
        """
        F = self.cfg.n_features
        if x.size(-1) != F:
            raise ValueError(
                f"Expected last dim {F}, got {x.size(-1)} (shape={tuple(x.shape)})"
            )

        # Get feature-index positional embeddings: (F, hidden_dim)
        idx = torch.arange(F, device=x.device)
        pos = self.feature_pos_embed(idx)  # (F, d)

        # Combine x and pos: for each feature dim d, concat (x[:, d:d+1], pos[d])
        # Reshape x to (..., n_firms, F, 1)
        x_per_dim = x.unsqueeze(-1)  # (..., n_firms, F, 1)

        # Broadcast pos to match: (1, ..., 1, F, hidden_dim) over leading dims
        # We'll use broadcasting; pos has shape (F, d).
        # Construct shape: (*1s, F, d) then broadcast.
        broadcast_shape = (1,) * (x.ndim - 1) + (F, self.cfg.hidden_dim)
        pos_b = pos.view(broadcast_shape)  # (1,...,1, F, d)
        pos_b = pos_b.expand(*x.shape[:-1], F, self.cfg.hidden_dim)

        combined = torch.cat([x_per_dim, pos_b], dim=-1)  # (..., n_firms, F, 1+d)

        # Apply phi to each (firm, feature_idx) row.
        phi_out = self.phi(combined)  # (..., n_firms, F, d)

        # Pool across feature dim.
        if self.pool_kind == "mean":
            h = phi_out.mean(dim=-2)
        elif self.pool_kind == "sum":
            h = phi_out.sum(dim=-2)
        else:
            raise ValueError(f"Unknown pool kind: {self.pool_kind}")

        return h


# =============================================================================
# Hypergraph convolution (SC-RIHN Eq. 3 / Feng et al. 2019).
# =============================================================================


class HypergraphConv(nn.Module):
    """One layer of hypergraph convolution.

    Implements:
        Z' = sigma( D_v^(-1/2) H W_e D_e^(-1) H^T D_v^(-1/2) Z W^(l) )

    where:
        H   : (n_firms, n_he)  incidence matrix
        W_e : (n_he, n_he)     diagonal hyperedge weights (init: identity)
        D_v : (n_firms, n_firms) diag of vertex degrees in hypergraph
        D_e : (n_he, n_he)       diag of hyperedge sizes
        W^(l): (d, d)            learnable layer weight

    For efficiency we precompute the normalization combinations on first
    `set_hypergraph()` call. The forward pass batches over leading dims so
    the same layer can be applied across timesteps in a single matmul.

    Input shape (after `set_hypergraph(H)`):
        Z : (..., n_firms, d_in)
    Output shape:
        Z' : (..., n_firms, d_out)
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 use_layer_norm: bool = True, activation: str = "gelu") -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()
        self.act = _make_activation(activation)

        # Normalization buffer; set via `set_hypergraph`.
        self.register_buffer("propagation", None, persistent=False)
        # Stored hyperedge weight diagonal; init lazily.
        self._n_he: Optional[int] = None
        self.W_e_diag: Optional[nn.Parameter] = None

    def set_hypergraph(
        self,
        H: torch.Tensor,
        learnable_edge_weights: bool = False,
    ) -> None:
        """Precompute the normalization matrix from incidence H.

        Call this once when the hypergraph is set; the layer caches the
        normalization. Must be called BEFORE forward().

        Parameters
        ----------
        H : torch.Tensor, shape (n_firms, n_he)
            Incidence matrix.
        learnable_edge_weights : bool
            If True, registers W_e_diag as a learnable parameter. Default
            False (SC-RIHN's choice: equal weights).
        """
        if H.ndim != 2:
            raise ValueError(f"H must be 2-D; got shape {tuple(H.shape)}.")
        device = H.device

        n_firms, n_he = H.shape
        self._n_he = n_he

        # Vertex degrees: D_v[i,i] = sum over hyperedges of H[i,e] * W_e[e,e].
        # Edge degrees:   D_e[e,e] = sum over vertices of H[v,e].
        # Equal weights initially -> W_e = I, so D_v[i] = row sum of H.
        D_v = H.sum(dim=1)  # (n_firms,)
        D_e = H.sum(dim=0)  # (n_he,)

        # Avoid division by zero — both should be > 0 in valid hypergraphs.
        D_v_inv_sqrt = torch.where(
            D_v > 0,
            D_v.pow(-0.5),
            torch.zeros_like(D_v),
        )
        D_e_inv = torch.where(
            D_e > 0,
            D_e.pow(-1.0),
            torch.zeros_like(D_e),
        )

        # Propagation matrix: D_v^(-1/2) H D_e^(-1) H^T D_v^(-1/2)
        # (W_e = I here; learnable W_e applied at forward time if enabled.)
        H_left = D_v_inv_sqrt.unsqueeze(1) * H            # (n_firms, n_he)
        H_right = H * D_e_inv.unsqueeze(0)                # (n_firms, n_he)
        # Final propagation: H_left @ (H_right.T @ D_v_inv_sqrt-scaled)
        # = (D_v^(-1/2) H D_e^(-1)) @ (H^T D_v^(-1/2))
        H_right = H_right.T * D_v_inv_sqrt.unsqueeze(0)   # (n_he, n_firms)
        propagation = H_left @ H_right                    # (n_firms, n_firms)

        self.propagation = propagation.to(device)

        if learnable_edge_weights:
            self.W_e_diag = nn.Parameter(torch.ones(n_he, device=device))
        else:
            self.W_e_diag = None

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Apply one hypergraph convolution.

        Parameters
        ----------
        Z : torch.Tensor, shape (..., n_firms, in_dim)

        Returns
        -------
        Z' : torch.Tensor, shape (..., n_firms, out_dim)
        """
        if self.propagation is None:
            raise RuntimeError(
                "HypergraphConv: must call set_hypergraph(H) before forward()."
            )

        # If learnable edge weights are enabled, we cannot use the precomputed
        # propagation directly; recompute it lazily here.
        if self.W_e_diag is not None:
            # Fall back to recomputation each forward; this is the cost of
            # learnable edge weights and is ablated only.
            raise NotImplementedError(
                "Learnable W_e not yet plumbed in fast path. "
                "Plumb by recomputing propagation in forward()."
            )

        # Z: (..., n_firms, in_dim).
        # Apply weight matrix first (smaller cost when out_dim < n_firms).
        ZW = self.weight(Z)  # (..., n_firms, out_dim)
        ZW = self.dropout(ZW)

        # Propagate via incidence-induced matrix: (..., n_firms, out_dim).
        # propagation: (n_firms, n_firms). Apply along the n_firms axis.
        # Use einsum for clarity and to handle arbitrary leading dims.
        out = torch.einsum("ij,...jk->...ik", self.propagation, ZW)

        out = self.norm(out)
        out = self.act(out)
        return out


# =============================================================================
# Temporal block: TCN over the time dimension.
# =============================================================================


class CausalConv1d(nn.Module):
    """1-D causal convolution with dilation.

    Pads the left side so that the output at time t only depends on inputs
    at times <= t. Standard component in TCNs (Bai et al. 2018).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., channels, time)
        x = nn.functional.pad(x, (self.left_pad, 0))
        return self.conv(x)


class MeanPoolTemporal(nn.Module):
    """SC-RIHN's mean-pool temporal readout (faithful replication baseline).

    Simply averages per-step embeddings across the T axis. Used by
    `replicate_sc_rihn.py` to verify our encoder matches SC-RIHN's
    published F1 within tolerance.

    Input shape:  (..., n_firms, T, d)
    Output shape: (..., n_firms, d)
    """
    def forward(self, Z_seq: torch.Tensor) -> torch.Tensor:
        return Z_seq.mean(dim=-2)


class TemporalConvBlock(nn.Module):
    """TCN over the time dimension, applied independently per firm.

    Layer k uses dilation 2^k (so 3 layers cover receptive field 1+2+4 = 7
    with kernel=3). This is enough to cover T=5 timesteps end-to-end.

    Input shape:  (..., n_firms, T, d)
    Output shape: (..., n_firms, d)  -- after temporal pooling
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        for k in range(n_layers):
            self.layers.append(
                CausalConv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2 ** k,
                )
            )
        self.act = _make_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, Z_seq: torch.Tensor) -> torch.Tensor:
        """Apply TCN over time.

        Parameters
        ----------
        Z_seq : torch.Tensor, shape (..., n_firms, T, d)

        Returns
        -------
        out : torch.Tensor, shape (..., n_firms, d)
            Last-timestep activation after the TCN (causal so this carries
            the full receptive-field context).
        """
        # We need (..., d, T) for Conv1d. Combine all batch dims (including
        # firms) into a single batch axis.
        orig_shape = Z_seq.shape  # (..., n_firms, T, d)
        T = orig_shape[-2]
        d = orig_shape[-1]
        if d != self.hidden_dim:
            raise ValueError(
                f"TCN expects last dim {self.hidden_dim}, got {d}"
            )

        batch_size = int(torch.tensor(orig_shape[:-2]).prod().item())
        # (B, T, d) -> (B, d, T) for Conv1d.
        x = Z_seq.reshape(batch_size, T, d).transpose(1, 2)  # (B, d, T)

        for layer in self.layers:
            residual = x
            x = layer(x)              # (B, d, T)
            x = self.act(x)
            x = self.dropout(x)
            x = x + residual          # residual connection

        # Take the last timestep (causal so this is full-context).
        out = x[..., -1]  # (B, d)
        out = self.final_norm(out)

        # Reshape back to leading dims minus the T axis.
        out = out.reshape(*orig_shape[:-2], d)
        return out


# =============================================================================
# Full HyperCP encoder.
# =============================================================================


class HyperCPEncoder(nn.Module):
    """The shared encoder: DeepSets -> [HGNN x L] -> TCN.

    Inputs
    ------
    features : (n_firms, T, F) or (B, n_firms, T, F)
        Multi-channel temporal features per firm.
    incidence : (n_firms, n_he)
        Set once via `set_hypergraph(H)`.

    Outputs
    -------
    z_seq : (n_firms, T, d) or (B, n_firms, T, d)
        Per-firm, per-timestep structural embedding (after HGNN, before TCN).
        Used by the forecast and quantile heads (they consume per-timestep
        context).
    z_final : (n_firms, d) or (B, n_firms, d)
        Per-firm summary after TCN, last-timestep activation. Used by the
        resilience functional and as a system-level summary.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Feature encoder (DeepSets).
        self.feature_encoder = DeepSetsFeatureEncoder(cfg)

        # Stack of L HGNN convolutions.
        self.hgnn_layers = nn.ModuleList([
            HypergraphConv(
                in_dim=cfg.hidden_dim,
                out_dim=cfg.hidden_dim,
                dropout=cfg.dropout,
                use_layer_norm=cfg.use_layer_norm,
                activation=cfg.activation,
            )
            for _ in range(cfg.n_hgnn_layers)
        ])

        # Temporal readout: TCN (ours) or simple mean pool (SC-RIHN faithful).
        if cfg.temporal_readout == "tcn":
            self.temporal = TemporalConvBlock(
                hidden_dim=cfg.hidden_dim,
                n_layers=cfg.n_tcn_layers,
                kernel_size=cfg.tcn_kernel,
                dropout=cfg.dropout,
                activation=cfg.activation,
            )
        elif cfg.temporal_readout == "mean_pool":
            self.temporal = MeanPoolTemporal()
        else:
            raise ValueError(
                f"Unknown temporal_readout '{cfg.temporal_readout}'. "
                f"Options: 'tcn', 'mean_pool'."
            )

        self._hypergraph_set = False

    # ---------------------------------------------------------------------
    # Hypergraph wiring.
    # ---------------------------------------------------------------------

    def set_hypergraph(
        self,
        incidence: torch.Tensor,
        learnable_edge_weights: bool = False,
    ) -> None:
        """Wire the hypergraph into every HGNN layer.

        Call once after instantiation, before any forward pass. Re-call if
        the hypergraph changes (e.g. SCR-NR / SCR-ER perturbation variants).
        """
        for layer in self.hgnn_layers:
            layer.set_hypergraph(incidence, learnable_edge_weights)
        self._hypergraph_set = True

    # ---------------------------------------------------------------------
    # Forward.
    # ---------------------------------------------------------------------

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-firm structural embeddings.

        Parameters
        ----------
        features : torch.Tensor, shape (n_firms, T, F) or (B, n_firms, T, F)

        Returns
        -------
        z_seq : torch.Tensor, shape (..., n_firms, T, d)
            Per-timestep embeddings after HGNN stack.
        z_final : torch.Tensor, shape (..., n_firms, d)
            Final per-firm summary after TCN.
        """
        if not self._hypergraph_set:
            raise RuntimeError(
                "HyperCPEncoder: call set_hypergraph(H) before forward()."
            )

        # features: (..., n_firms, T, F). We need DeepSets per (firm, t).
        # Move T axis to position before n_firms to apply DeepSets cleanly.
        # Easiest: combine all leading dims + T as outer batch.
        ndim = features.ndim
        if ndim == 3:
            # (n_firms, T, F) -- single sample
            features = features.unsqueeze(0)  # (1, n_firms, T, F)
            squeeze_batch = True
        else:
            squeeze_batch = False

        # Shape now: (B, n_firms, T, F).
        B, n_firms, T, F = features.shape

        # 1. DeepSets per (firm, t).
        # Permute T to before firms so DeepSets sees (B, T, n_firms, F).
        x = features.permute(0, 2, 1, 3)             # (B, T, n_firms, F)
        h = self.feature_encoder(x)                  # (B, T, n_firms, d)

        # 2. HGNN stack — applied per timestep.
        # For each timestep, propagate through L HGNN layers.
        # Vectorise over (B, T) by treating it as a single batch axis.
        # h: (B, T, n_firms, d).
        z = h
        for layer in self.hgnn_layers:
            z = layer(z)                              # (B, T, n_firms, d)

        # Permute back to (B, n_firms, T, d) for the temporal block.
        z_seq = z.permute(0, 2, 1, 3)                # (B, n_firms, T, d)

        # 3. TCN temporal block.
        z_final = self.temporal(z_seq)               # (B, n_firms, d)

        if squeeze_batch:
            z_seq = z_seq.squeeze(0)
            z_final = z_final.squeeze(0)
        return z_seq, z_final

    # ---------------------------------------------------------------------
    # Param-count utility for logging.
    # ---------------------------------------------------------------------

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Smoke test.
# =============================================================================


def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    torch.manual_seed(0)

    # --- Setup: tiny network for fast CPU test ---
    n_firms = 10
    T = 5
    F = 5
    n_he = 8

    cfg = EncoderConfig(
        n_features=F,
        hidden_dim=16,        # smaller for smoke test
        n_hgnn_layers=4,      # same depth
        n_tcn_layers=3,
        tcn_kernel=3,
        dropout=0.1,
    )
    print(f"Config: {cfg}")

    encoder = HyperCPEncoder(cfg)
    print(f"Encoder param count: {encoder.n_parameters():,}")

    # --- Fake hypergraph: 10 firms, 8 hyperedges, sparse incidence ---
    incidence = (torch.rand(n_firms, n_he) > 0.5).float()
    # Ensure no empty rows or columns (degenerate degree matrices).
    incidence[0, 0] = 1.0
    for i in range(n_he):
        if incidence[:, i].sum() < 2:
            incidence[:2, i] = 1.0
    for c in range(n_firms):
        if incidence[c, :].sum() < 1:
            incidence[c, 0] = 1.0

    encoder.set_hypergraph(incidence)
    print(f"Incidence shape: {tuple(incidence.shape)}, "
          f"density: {incidence.mean().item():.3f}")

    # --- Forward pass: unbatched ---
    features = torch.randn(n_firms, T, F)
    z_seq, z_final = encoder(features)
    print(f"\nUnbatched forward:")
    print(f"  input  features:  {tuple(features.shape)}")
    print(f"  output z_seq:     {tuple(z_seq.shape)}")
    print(f"  output z_final:   {tuple(z_final.shape)}")
    assert z_seq.shape == (n_firms, T, cfg.hidden_dim)
    assert z_final.shape == (n_firms, cfg.hidden_dim)
    print(f"  [OK] shapes match expected.")

    # --- Forward pass: batched ---
    B = 4
    features_b = torch.randn(B, n_firms, T, F)
    z_seq_b, z_final_b = encoder(features_b)
    print(f"\nBatched forward (B={B}):")
    print(f"  input features:   {tuple(features_b.shape)}")
    print(f"  output z_seq:     {tuple(z_seq_b.shape)}")
    print(f"  output z_final:   {tuple(z_final_b.shape)}")
    assert z_seq_b.shape == (B, n_firms, T, cfg.hidden_dim)
    assert z_final_b.shape == (B, n_firms, cfg.hidden_dim)
    print(f"  [OK] shapes match expected.")

    # --- Gradient flow check ---
    loss = z_final.pow(2).mean()
    loss.backward()
    has_grad = sum(1 for p in encoder.parameters() if p.grad is not None
                   and p.grad.abs().sum().item() > 0)
    total = sum(1 for p in encoder.parameters() if p.requires_grad)
    print(f"\nGradient flow: {has_grad}/{total} parameters have nonzero grad.")
    assert has_grad == total, "Some parameters did not receive gradient."
    print("  [OK] all parameters receive gradient.")

    # --- Device portability check ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        encoder_gpu = HyperCPEncoder(cfg).to(device)
        encoder_gpu.set_hypergraph(incidence.to(device))
        features_gpu = features.to(device)
        z_seq_g, z_final_g = encoder_gpu(features_gpu)
        print(f"\n[OK] GPU forward: z_final on {z_final_g.device}, "
              f"shape {tuple(z_final_g.shape)}")
    else:
        print(f"\nGPU not available; CPU smoke test complete.")

    # --- End-to-end with real SupplyGraph ---
    print("\n" + "=" * 60)
    print("End-to-end smoke test against real SupplyGraph data:")
    print("=" * 60)
    from hypercp.data.supplygraph import SupplyGraphHypergraph
    sg = SupplyGraphHypergraph()
    real_incidence = sg.hyperedge_incidence_matrix()
    cfg_real = EncoderConfig(
        n_features=sg.n_channels,
        hidden_dim=64,        # SC-RIHN's reported optimum
        n_hgnn_layers=4,
        n_tcn_layers=3,
    )
    encoder_real = HyperCPEncoder(cfg_real)
    encoder_real.set_hypergraph(real_incidence)

    # Use T=5 rolling window from the data.
    T_window = 5
    # Pick a random starting day in the train range.
    start = 100
    feat = sg.node_features[:, start:start + T_window, :]  # (n_products, T, F)
    z_seq_real, z_final_real = encoder_real(feat)
    print(f"SupplyGraph features:    {tuple(feat.shape)}")
    print(f"SupplyGraph z_seq:       {tuple(z_seq_real.shape)}")
    print(f"SupplyGraph z_final:     {tuple(z_final_real.shape)}")
    print(f"Encoder param count:     {encoder_real.n_parameters():,}")
    print("  [OK] encoder consumes real SupplyGraph data without modification.")


if __name__ == "__main__":
    _smoke_test()
