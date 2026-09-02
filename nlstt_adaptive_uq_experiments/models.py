from __future__ import annotations

import torch
from torch import nn


class SparseTrajectoryEncoder(nn.Module):
    """Encode sparse observed times and log-volumes into target log-volume."""

    def __init__(self, max_points: int = 8, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.max_points = max_points
        in_dim = max_points * 2 + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 11),
        )

    def forward(self, t_obs: torch.Tensor, y_obs: torch.Tensor, mask: torch.Tensor, t_target: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([t_obs * mask, y_obs * mask, t_target[:, None]], dim=1)
        raw = self.net(x)
        y_pred = raw[:, 0]
        alpha = torch.nn.functional.softplus(raw[:, 1]) + 1e-4
        log_k = raw[:, 2]
        rate = raw[:, 3]
        decay = torch.nn.functional.softplus(raw[:, 4]) + 1e-4
        mix_logits = raw[:, 5:10]
        mix_probs = torch.nn.functional.softmax(mix_logits, dim=1)
        log_var = torch.clamp(raw[:, 10], min=-8.0, max=4.0)
        return {
            "logv_pred": y_pred,
            "log_var": log_var,
            "alpha": alpha,
            "log_k": log_k,
            "rate": rate,
            "decay": decay,
            "mix_logits": mix_logits,
            "mix_probs": mix_probs,
        }


def _trajectory_features(t_obs: torch.Tensor, y_obs: torch.Tensor, mask: torch.Tensor, t_target: torch.Tensor) -> torch.Tensor:
    n = y_obs.shape[0]
    lengths = torch.clamp(mask.sum(dim=1).long(), min=1)
    first_y = y_obs[:, 0]
    first_t = t_obs[:, 0]
    last_idx = lengths - 1
    row_idx = torch.arange(n, device=y_obs.device)
    last_y = y_obs[row_idx, last_idx]
    last_t = t_obs[row_idx, last_idx]
    prev_idx = torch.clamp(lengths - 2, min=0)
    prev_y = y_obs[row_idx, prev_idx]
    prev_t = t_obs[row_idx, prev_idx]
    dt_last = torch.clamp(last_t - prev_t, min=1e-6)
    dt_global = torch.clamp(last_t - first_t, min=1e-6)
    target_dt = torch.clamp(t_target - last_t, min=1e-6)
    local_slope = torch.where(lengths > 1, (last_y - prev_y) / dt_last, torch.zeros_like(last_y))
    global_slope = torch.where(lengths > 1, (last_y - first_y) / dt_global, torch.zeros_like(last_y))
    valid_count = torch.clamp(mask.sum(dim=1), min=1.0)
    mean_y = (y_obs * mask).sum(dim=1) / valid_count
    centered = (y_obs - mean_y[:, None]) * mask
    std_y = torch.sqrt(torch.sum(centered**2, dim=1) / valid_count + 1e-6)
    linear_local = last_y + local_slope * target_dt
    linear_global = last_y + global_slope * target_dt
    stable = last_y
    shrink = last_y - torch.abs(local_slope) * target_dt
    return torch.stack(
        [
            valid_count / 4.0,
            first_y,
            last_y,
            last_y - first_y,
            mean_y,
            std_y,
            local_slope,
            global_slope,
            target_dt,
            dt_global,
            stable,
            linear_local,
            linear_global,
            shrink,
        ],
        dim=1,
    )


class PhysicsFeatureFusionEncoder(nn.Module):
    """Fuse sparse trajectory values with explicit physics-inspired extrapolation features."""

    def __init__(self, max_points: int = 8, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.max_points = max_points
        in_dim = max_points * 2 + 1 + 14
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, t_obs: torch.Tensor, y_obs: torch.Tensor, mask: torch.Tensor, t_target: torch.Tensor) -> dict[str, torch.Tensor]:
        base = torch.cat([t_obs * mask, y_obs * mask, t_target[:, None]], dim=1)
        feats = _trajectory_features(t_obs, y_obs, mask, t_target)
        raw = self.net(torch.cat([base, feats], dim=1))
        y_pred = raw[:, 0]
        alpha = torch.nn.functional.softplus(raw[:, 1]) + 1e-4
        log_k = raw[:, 2]
        rate = raw[:, 3]
        decay = torch.nn.functional.softplus(raw[:, 4]) + 1e-4
        log_var = torch.clamp(raw[:, 5], min=-8.0, max=4.0)
        mix_probs = torch.zeros(y_pred.shape[0], 5, dtype=y_pred.dtype, device=y_pred.device)
        mix_probs[:, 2] = 1.0
        return {
            "logv_pred": y_pred,
            "log_var": log_var,
            "alpha": alpha,
            "log_k": log_k,
            "rate": rate,
            "decay": decay,
            "mix_logits": torch.zeros_like(mix_probs),
            "mix_probs": mix_probs,
        }


class AnchoredCorrectionEncoder(nn.Module):
    """Predict a bounded correction away from the most recent observed CT value."""

    def __init__(self, max_points: int = 8, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.max_points = max_points
        in_dim = max_points * 2 + 1 + 14
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 7),
        )

    def forward(self, t_obs: torch.Tensor, y_obs: torch.Tensor, mask: torch.Tensor, t_target: torch.Tensor) -> dict[str, torch.Tensor]:
        base = torch.cat([t_obs * mask, y_obs * mask, t_target[:, None]], dim=1)
        feats = _trajectory_features(t_obs, y_obs, mask, t_target)
        raw = self.net(torch.cat([base, feats], dim=1))
        lengths = torch.clamp(mask.sum(dim=1).long(), min=1)
        row_idx = torch.arange(y_obs.shape[0], device=y_obs.device)
        last_y = y_obs[row_idx, lengths - 1]
        correction_gate = torch.sigmoid(raw[:, 0])
        correction = raw[:, 1]
        y_pred = last_y + correction_gate * correction
        alpha = torch.nn.functional.softplus(raw[:, 2]) + 1e-4
        log_k = raw[:, 3]
        rate = raw[:, 4]
        decay = torch.nn.functional.softplus(raw[:, 5]) + 1e-4
        log_var = torch.clamp(raw[:, 6], min=-8.0, max=4.0)
        mix_probs = torch.zeros(y_pred.shape[0], 5, dtype=y_pred.dtype, device=y_pred.device)
        mix_probs[:, 4] = 1.0
        return {
            "logv_pred": y_pred,
            "log_var": log_var,
            "alpha": alpha,
            "log_k": log_k,
            "rate": rate,
            "decay": decay,
            "correction_gate": correction_gate,
            "mix_logits": torch.zeros_like(mix_probs),
            "mix_probs": mix_probs,
        }


def pad_batch(batch: list[dict], max_points: int = 8) -> dict[str, torch.Tensor | list[str]]:
    n = len(batch)
    t_obs = torch.zeros(n, max_points, dtype=torch.float32)
    y_obs = torch.zeros(n, max_points, dtype=torch.float32)
    mask = torch.zeros(n, max_points, dtype=torch.float32)
    t_target = torch.zeros(n, dtype=torch.float32)
    y_target = torch.zeros(n, dtype=torch.float32)
    ids = []
    patient_ids = []
    for i, item in enumerate(batch):
        t = torch.tensor(item["t_obs"], dtype=torch.float32)
        y = torch.tensor(item["logv_obs"], dtype=torch.float32)
        k = min(len(t), max_points)
        t_obs[i, :k] = t[:k]
        y_obs[i, :k] = y[:k]
        mask[i, :k] = 1.0
        t_target[i] = float(item["t_target"])
        y_target[i] = float(item["logv_target"])
        ids.append(str(item["trajectory_id"]))
        patient_ids.append(str(item.get("patient_id", item["trajectory_id"])))
    return {
        "t_obs": t_obs,
        "y_obs": y_obs,
        "mask": mask,
        "t_target": t_target,
        "y_target": y_target,
        "ids": ids,
        "patient_ids": patient_ids,
    }
