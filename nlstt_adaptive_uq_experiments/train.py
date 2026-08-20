from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .models import AnchoredCorrectionEncoder, PhysicsFeatureFusionEncoder, SparseTrajectoryEncoder, pad_batch


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.records = frame.to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


@dataclass
class TrainedModel:
    model: SparseTrajectoryEncoder
    config: TrainConfig
    method: str
    lambda_log: pd.DataFrame | None = None


MIXTURE_PHYSICS_NAMES = ["gompertz", "logistic", "exponential", "decay", "stable"]
HETEROSCEDASTIC_METHODS = {"heteroscedastic", "heteroscedastic_ensemble"}


def _lambda_from_uncertainty(pred_std: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    lam = cfg.fixed_lambda_phys * (1.0 + cfg.adaptive_beta * pred_std.detach())
    return torch.clamp(lam, cfg.adaptive_lambda_min, cfg.adaptive_lambda_max)


def _lambda_inverse_uncertainty(pred_std: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    lam = cfg.fixed_lambda_phys / (1.0 + cfg.adaptive_beta * pred_std.detach())
    return torch.clamp(lam, cfg.adaptive_lambda_min, cfg.adaptive_lambda_max)


def _lambda_epoch(epoch: int, cfg: TrainConfig) -> torch.Tensor:
    progress = epoch / max(cfg.epochs - 1, 1)
    lam = cfg.adaptive_lambda_min * (cfg.adaptive_lambda_max / cfg.adaptive_lambda_min) ** progress
    return torch.tensor(float(lam), dtype=torch.float32)


def _safe_nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _lambda_reliability_gated(
    pred_std: torch.Tensor,
    phys_residual: torch.Tensor,
    batch: dict,
    cfg: TrainConfig,
    mode: str = "full",
) -> tuple[torch.Tensor, dict[str, float]]:
    uncertainty = pred_std.detach()
    one = torch.tensor(1.0, dtype=torch.float32, device=pred_std.device)
    r_uq = torch.exp(-cfg.reliability_uq_beta * uncertainty)
    m_obs = batch["mask"].sum(dim=1).detach()
    r_sparse = torch.clamp((m_obs / 4.0) ** cfg.reliability_sparsity_power, 0.0, 1.0).mean()
    r_phys = torch.exp(-cfg.reliability_phys_gamma * torch.mean(torch.abs(phys_residual.detach())))
    if mode == "uq_only":
        r_sparse = one
        r_phys = one
    elif mode == "sparsity_only":
        r_uq = one
        r_phys = one
    elif mode == "phys_only":
        r_uq = one
        r_sparse = one
    elif mode == "boosted":
        r_uq = 1.0 - r_uq
    reliability = torch.clamp(r_uq * r_sparse * r_phys, 0.0, 1.0)
    lam = cfg.adaptive_lambda_min + (cfg.adaptive_lambda_max - cfg.adaptive_lambda_min) * reliability
    return lam, {
        "reliability": float(reliability.detach().cpu().numpy()),
        "r_uq": float(r_uq.detach().cpu().numpy()),
        "r_sparse": float(r_sparse.detach().cpu().numpy()),
        "r_phys": float(r_phys.detach().cpu().numpy()),
    }


def _last_observation_residual_terms(out: dict[str, torch.Tensor], batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    last_idx = torch.clamp(batch["mask"].sum(dim=1).long() - 1, min=0)
    last_y = batch["y_obs"][torch.arange(batch["y_obs"].shape[0]), last_idx]
    last_t = batch["t_obs"][torch.arange(batch["t_obs"].shape[0]), last_idx]
    dt = torch.clamp(batch["t_target"] - last_t, min=1e-6)
    dy_dt = (out["logv_pred"] - last_y) / dt
    return last_y, dy_dt


def _gompertz_residual(out: dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
    last_y, dy_dt = _last_observation_residual_terms(out, batch)
    rhs = out["alpha"] * (out["log_k"] - 0.5 * (last_y + out["logv_pred"]))
    return dy_dt - rhs


def _mixture_physics_residuals(out: dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
    last_y, dy_dt = _last_observation_residual_terms(out, batch)
    y_mid = 0.5 * (last_y + out["logv_pred"])
    k = torch.exp(torch.clamp(out["log_k"], min=-8.0, max=8.0))
    v_mid = torch.exp(torch.clamp(y_mid, min=-8.0, max=8.0))
    gompertz_rhs = out["alpha"] * (out["log_k"] - y_mid)
    logistic_rhs = out["alpha"] * (1.0 - v_mid / torch.clamp(k, min=1e-4))
    exponential_rhs = out["rate"]
    decay_rhs = -out["decay"]
    stable_rhs = torch.zeros_like(dy_dt)
    rhs = torch.stack([gompertz_rhs, logistic_rhs, exponential_rhs, decay_rhs, stable_rhs], dim=1)
    return dy_dt[:, None] - rhs


def _mixture_physics_loss(out: dict[str, torch.Tensor], batch: dict, cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    residuals = _mixture_physics_residuals(out, batch)
    probs = out["mix_probs"]
    weighted_residual = torch.sum(probs * residuals, dim=1)
    residual_loss = torch.mean(torch.sum(probs * residuals**2, dim=1))
    entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, min=1e-8)), dim=1))
    return residual_loss + cfg.mixture_entropy_weight * entropy, weighted_residual


def _heteroscedastic_gaussian_nll(out: dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
    log_var = torch.clamp(out["log_var"], min=-8.0, max=4.0)
    inv_var = torch.exp(-log_var)
    sq_err = (out["logv_pred"] - batch["y_target"]) ** 2
    return torch.mean(0.5 * (log_var + sq_err * inv_var))


def train_single_model(train_df: pd.DataFrame, cfg: TrainConfig, method: str = "fixed_pinn", seed: int = 0) -> TrainedModel:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    reliability_methods = {
        "reliability_gated_pinn",
        "reliability_uq_only_pinn",
        "reliability_sparsity_only_pinn",
        "reliability_phys_only_pinn",
        "reliability_boosted_pinn",
    }
    mixture_methods = {"mixture_physics_pinn", "mixture_physics_uq"}
    fusion_methods = {"physics_feature_fusion", "physics_feature_fusion_uq"}
    anchor_methods = {"anchored_correction", "anchored_correction_uq", "anchored_correction_pinn"}
    stochastic_methods = {
        "mc_dropout",
        "fixed_pinn_uq",
        "adaptive_pinn",
        "inverse_adaptive_pinn",
        "mixture_physics_uq",
        "physics_feature_fusion_uq",
        "anchored_correction_uq",
    }
    dropout = cfg.dropout if method in stochastic_methods | reliability_methods else 0.0
    if method in anchor_methods:
        model = AnchoredCorrectionEncoder(hidden_dim=cfg.hidden_dim, dropout=dropout)
    elif method in fusion_methods:
        model = PhysicsFeatureFusionEncoder(hidden_dim=cfg.hidden_dim, dropout=dropout)
    else:
        model = SparseTrajectoryEncoder(hidden_dim=cfg.hidden_dim, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(
        TrajectoryDataset(train_df),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: pad_batch(b, max_points=8),
    )
    model.train()
    lambda_records = []
    for epoch in range(cfg.epochs):
        epoch_lams = []
        epoch_uncertainty = []
        epoch_data_losses = []
        epoch_phys_losses = []
        epoch_reliability = []
        epoch_r_uq = []
        epoch_r_sparse = []
        epoch_r_phys = []
        for batch in loader:
            out = model(batch["t_obs"], batch["y_obs"], batch["mask"], batch["t_target"])
            if method in HETEROSCEDASTIC_METHODS:
                data_loss = _heteroscedastic_gaussian_nll(out, batch)
            else:
                data_loss = torch.mean((out["logv_pred"] - batch["y_target"]) ** 2)
            if method in mixture_methods:
                phys_loss, phys_residual = _mixture_physics_loss(out, batch, cfg)
            else:
                phys_residual = _gompertz_residual(out, batch)
                phys_loss = torch.mean(phys_residual**2)
            pred_std = torch.std(out["logv_pred"]).reshape(())
            reliability_parts = {"reliability": np.nan, "r_uq": np.nan, "r_sparse": np.nan, "r_phys": np.nan}
            if method in {"no_physics"} | HETEROSCEDASTIC_METHODS:
                lam = 0.0
            elif method == "adaptive_pinn":
                # During training use batch residual dispersion as a cheap uncertainty proxy.
                lam = _lambda_from_uncertainty(pred_std, cfg)
            elif method == "inverse_adaptive_pinn":
                lam = _lambda_inverse_uncertainty(pred_std, cfg)
            elif method in reliability_methods:
                gate_mode = {
                    "reliability_gated_pinn": "full",
                    "reliability_uq_only_pinn": "uq_only",
                    "reliability_sparsity_only_pinn": "sparsity_only",
                    "reliability_phys_only_pinn": "phys_only",
                    "reliability_boosted_pinn": "boosted",
                }[method]
                lam, reliability_parts = _lambda_reliability_gated(pred_std, phys_residual, batch, cfg, mode=gate_mode)
            elif method == "epoch_pinn":
                lam = _lambda_epoch(epoch, cfg)
            elif method == "random_pinn":
                lam = torch.tensor(float(rng.uniform(cfg.adaptive_lambda_min, cfg.adaptive_lambda_max)), dtype=torch.float32)
            elif method in mixture_methods:
                lam = cfg.fixed_lambda_phys
            elif method in fusion_methods:
                lam = cfg.fixed_lambda_phys
            elif method in {"anchored_correction", "anchored_correction_uq"}:
                lam = 0.0
            elif method == "anchored_correction_pinn":
                lam = cfg.fixed_lambda_phys
            else:
                lam = cfg.fixed_lambda_phys
            loss = data_loss + lam * phys_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            lam_float = float(lam.detach().cpu().numpy()) if isinstance(lam, torch.Tensor) else float(lam)
            epoch_lams.append(lam_float)
            epoch_uncertainty.append(float(pred_std.detach().cpu().numpy()))
            epoch_data_losses.append(float(data_loss.detach().cpu().numpy()))
            epoch_phys_losses.append(float(phys_loss.detach().cpu().numpy()))
            epoch_reliability.append(reliability_parts["reliability"])
            epoch_r_uq.append(reliability_parts["r_uq"])
            epoch_r_sparse.append(reliability_parts["r_sparse"])
            epoch_r_phys.append(reliability_parts["r_phys"])
        lambda_records.append(
            {
                "epoch": epoch + 1,
                "lambda_phys": float(np.mean(epoch_lams)) if epoch_lams else np.nan,
                "uncertainty_proxy": float(np.mean(epoch_uncertainty)) if epoch_uncertainty else np.nan,
                "data_loss": float(np.mean(epoch_data_losses)) if epoch_data_losses else np.nan,
                "physics_loss": float(np.mean(epoch_phys_losses)) if epoch_phys_losses else np.nan,
                "reliability": _safe_nanmean(epoch_reliability),
                "r_uq": _safe_nanmean(epoch_r_uq),
                "r_sparse": _safe_nanmean(epoch_r_sparse),
                "r_phys": _safe_nanmean(epoch_r_phys),
            }
        )
    return TrainedModel(model=model, config=cfg, method=method, lambda_log=pd.DataFrame(lambda_records))


def train_ensemble(
    train_df: pd.DataFrame,
    cfg: TrainConfig,
    seed: int = 0,
    method: str = "fixed_pinn",
) -> list[TrainedModel]:
    return [train_single_model(train_df, cfg, method=method, seed=seed + i) for i in range(cfg.ensemble_size)]
