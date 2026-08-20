from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .models import pad_batch
from .train import TrajectoryDataset


class LatentStateSpaceUQ(nn.Module):
    """Latent state-space predictor with separate process and measurement uncertainty."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, t_obs: torch.Tensor, y_obs: torch.Tensor, mask: torch.Tensor, t_target: torch.Tensor) -> dict[str, torch.Tensor]:
        n = y_obs.shape[0]
        h = torch.zeros(n, self.gru.hidden_size, dtype=y_obs.dtype, device=y_obs.device)
        prev_t = torch.zeros(n, dtype=y_obs.dtype, device=y_obs.device)
        for j in range(y_obs.shape[1]):
            active = mask[:, j] > 0.5
            dt = torch.clamp(t_obs[:, j] - prev_t, min=0.0)
            obs_feat = self.obs_encoder(torch.stack([y_obs[:, j], dt], dim=1))
            h_new = self.gru(obs_feat, h)
            h = torch.where(active[:, None], h_new, h)
            prev_t = torch.where(active, t_obs[:, j], prev_t)

        lengths = torch.clamp(mask.sum(dim=1).long(), min=1)
        row_idx = torch.arange(n, device=y_obs.device)
        last_y = y_obs[row_idx, lengths - 1]
        last_t = t_obs[row_idx, lengths - 1]
        first_y = y_obs[:, 0]
        target_dt = torch.clamp(t_target - last_t, min=1e-6)
        summary = torch.stack([last_y, last_y - first_y, target_dt], dim=1)
        raw = self.transition(torch.cat([h, summary], dim=1))
        delta = raw[:, 0]
        mean = last_y + delta
        process_var = torch.nn.functional.softplus(raw[:, 1]) + 1e-5
        measurement_var = torch.nn.functional.softplus(raw[:, 2]) + 1e-5
        alpha = torch.nn.functional.softplus(raw[:, 3]) + 1e-4
        log_k = raw[:, 4]
        total_var = process_var + measurement_var
        return {
            "logv_mean": mean,
            "process_var": process_var,
            "measurement_var": measurement_var,
            "total_var": total_var,
            "alpha": alpha,
            "log_k": log_k,
        }


@dataclass
class TrainedStateSpace:
    model: LatentStateSpaceUQ
    config: TrainConfig
    train_log: pd.DataFrame


def train_state_space_model(train_df: pd.DataFrame, cfg: TrainConfig, seed: int = 0) -> TrainedStateSpace:
    torch.manual_seed(seed)
    model = LatentStateSpaceUQ(hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(
        TrajectoryDataset(train_df),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: pad_batch(b, max_points=8),
    )
    records = []
    model.train()
    for epoch in range(cfg.epochs):
        losses = []
        nlls = []
        mses = []
        for batch in loader:
            out = model(batch["t_obs"], batch["y_obs"], batch["mask"], batch["t_target"])
            err = batch["y_target"] - out["logv_mean"]
            var = torch.clamp(out["total_var"], min=1e-5, max=25.0)
            nll = 0.5 * torch.mean(torch.log(var) + err**2 / var)
            mse = torch.mean(err**2)
            loss = nll + 0.05 * mse
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().numpy()))
            nlls.append(float(nll.detach().cpu().numpy()))
            mses.append(float(mse.detach().cpu().numpy()))
        records.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "nll": float(np.mean(nlls)),
                "mse": float(np.mean(mses)),
            }
        )
    return TrainedStateSpace(model=model, config=cfg, train_log=pd.DataFrame(records))


def predict_state_space(trained: TrainedStateSpace, frame: pd.DataFrame) -> pd.DataFrame:
    model = trained.model
    model.eval()
    loader = DataLoader(TrajectoryDataset(frame), batch_size=256, shuffle=False, collate_fn=lambda b: pad_batch(b, 8))
    rows = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["t_obs"], batch["y_obs"], batch["mask"], batch["t_target"])
            std = torch.sqrt(torch.clamp(out["total_var"], min=1e-5))
            for i, tid in enumerate(batch["ids"]):
                mean = float(out["logv_mean"][i])
                half = 1.959963984540054 * float(std[i])
                last_idx = int(torch.clamp(batch["mask"][i].sum().long() - 1, min=0).item())
                last_y = float(batch["y_obs"][i, last_idx])
                last_t = float(batch["t_obs"][i, last_idx])
                target_t = float(batch["t_target"][i])
                dt = max(target_t - last_t, 1e-6)
                dy_dt = (mean - last_y) / dt
                alpha = float(out["alpha"][i])
                log_k = float(out["log_k"][i])
                rhs = alpha * (log_k - 0.5 * (last_y + mean))
                physics_residual = dy_dt - rhs
                rows.append(
                    {
                        "trajectory_id": tid,
                        "logv_target": float(batch["y_target"][i]),
                        "logv_mean": mean,
                        "logv_lower": mean - half,
                        "logv_upper": mean + half,
                        "process_std": float(torch.sqrt(out["process_var"][i]).cpu().numpy()),
                        "measurement_std": float(torch.sqrt(out["measurement_var"][i]).cpu().numpy()),
                        "physics_residual": float(physics_residual),
                        "physics_residual_abs": float(abs(physics_residual)),
                    }
                )
    return pd.DataFrame(rows)
