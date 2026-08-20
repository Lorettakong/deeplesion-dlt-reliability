from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .models import pad_batch
from .train import HETEROSCEDASTIC_METHODS, MIXTURE_PHYSICS_NAMES, TrajectoryDataset, TrainedModel


def _predict_pass(trained: TrainedModel, frame: pd.DataFrame, train_mode: bool = False) -> pd.DataFrame:
    model = trained.model
    model.train(mode=train_mode)
    loader = DataLoader(TrajectoryDataset(frame), batch_size=256, shuffle=False, collate_fn=lambda b: pad_batch(b, 8))
    rows = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["t_obs"], batch["y_obs"], batch["mask"], batch["t_target"])
            for i, tid in enumerate(batch["ids"]):
                last_idx = int(torch.clamp(batch["mask"][i].sum().long() - 1, min=0).item())
                last_y = float(batch["y_obs"][i, last_idx])
                last_t = float(batch["t_obs"][i, last_idx])
                pred_y = float(out["logv_pred"][i])
                target_t = float(batch["t_target"][i])
                dt = max(target_t - last_t, 1e-6)
                dy_dt = (pred_y - last_y) / dt
                alpha = float(out["alpha"][i])
                log_k = float(out["log_k"][i])
                rhs = alpha * (log_k - 0.5 * (last_y + pred_y))
                physics_residual = dy_dt - rhs
                row = {
                    "trajectory_id": tid,
                    "logv_target": float(batch["y_target"][i]),
                    "logv_pred": pred_y,
                    "physics_residual": float(physics_residual),
                    "physics_residual_abs": float(abs(physics_residual)),
                }
                if trained.method in HETEROSCEDASTIC_METHODS and "log_var" in out:
                    row["log_var"] = float(out["log_var"][i])
                if "mix_probs" in out:
                    probs = out["mix_probs"][i].detach().cpu().numpy()
                    for name, prob in zip(MIXTURE_PHYSICS_NAMES, probs):
                        row[f"mix_{name}"] = float(prob)
                    row["mix_selected"] = MIXTURE_PHYSICS_NAMES[int(np.argmax(probs))]
                if "correction_gate" in out:
                    row["correction_gate"] = float(out["correction_gate"][i])
                rows.append(row)
    return pd.DataFrame(rows)


def predict_deterministic(trained: TrainedModel, frame: pd.DataFrame) -> pd.DataFrame:
    pred = _predict_pass(trained, frame, train_mode=False)
    pred["logv_mean"] = pred["logv_pred"]
    return pred.drop(columns=["logv_pred"])


def predict_mc_dropout(trained: TrainedModel, frame: pd.DataFrame, samples: int) -> pd.DataFrame:
    draws = [_predict_pass(trained, frame, train_mode=True).rename(columns={"logv_pred": f"draw_{i}"}) for i in range(samples)]
    merged = draws[0][["trajectory_id", "logv_target", "physics_residual", "physics_residual_abs"]].copy()
    for i, draw in enumerate(draws):
        merged = merged.merge(draw[["trajectory_id", f"draw_{i}"]], on="trajectory_id")
    draw_cols = [c for c in merged.columns if c.startswith("draw_")]
    vals = merged[draw_cols].to_numpy(float)
    merged["logv_mean"] = vals.mean(axis=1)
    merged["logv_lower"] = np.quantile(vals, 0.025, axis=1)
    merged["logv_upper"] = np.quantile(vals, 0.975, axis=1)
    return merged[["trajectory_id", "logv_target", "logv_mean", "logv_lower", "logv_upper", "physics_residual", "physics_residual_abs"]]


def predict_ensemble(models: list[TrainedModel], frame: pd.DataFrame) -> pd.DataFrame:
    draws = []
    for i, model in enumerate(models):
        draw = _predict_pass(model, frame).rename(columns={"logv_pred": f"draw_{i}"})
        if "log_var" in draw.columns:
            draw = draw.rename(columns={"log_var": f"log_var_{i}"})
        draws.append(draw)
    merged = draws[0][["trajectory_id", "logv_target", "physics_residual", "physics_residual_abs"]].copy()
    for i, draw in enumerate(draws):
        merged = merged.merge(draw[["trajectory_id", f"draw_{i}"]], on="trajectory_id")
        if f"log_var_{i}" in draw.columns:
            merged = merged.merge(draw[["trajectory_id", f"log_var_{i}"]], on="trajectory_id")
    draw_cols = [c for c in merged.columns if c.startswith("draw_")]
    vals = merged[draw_cols].to_numpy(float)
    merged["logv_mean"] = vals.mean(axis=1)
    log_var_cols = [c for c in merged.columns if c.startswith("log_var_")]
    if log_var_cols:
        aleatoric_var = np.exp(np.clip(merged[log_var_cols].to_numpy(float), -8.0, 4.0)).mean(axis=1)
        epistemic_var = vals.var(axis=1)
        total_std = np.sqrt(np.clip(aleatoric_var + epistemic_var, 1e-8, None))
        merged["logv_lower"] = merged["logv_mean"] - 1.959963984540054 * total_std
        merged["logv_upper"] = merged["logv_mean"] + 1.959963984540054 * total_std
        merged["aleatoric_std"] = np.sqrt(np.clip(aleatoric_var, 0.0, None))
        merged["epistemic_std"] = np.sqrt(np.clip(epistemic_var, 0.0, None))
        keep = [
            "trajectory_id",
            "logv_target",
            "logv_mean",
            "logv_lower",
            "logv_upper",
            "physics_residual",
            "physics_residual_abs",
            "aleatoric_std",
            "epistemic_std",
        ]
    else:
        merged["logv_lower"] = np.quantile(vals, 0.025, axis=1)
        merged["logv_upper"] = np.quantile(vals, 0.975, axis=1)
        keep = ["trajectory_id", "logv_target", "logv_mean", "logv_lower", "logv_upper", "physics_residual", "physics_residual_abs"]
    return merged[keep]


def estimate_residual_sigma(trained: TrainedModel, frame: pd.DataFrame) -> float:
    pred = _predict_pass(trained, frame, train_mode=False)
    resid = pred["logv_target"].to_numpy(float) - pred["logv_pred"].to_numpy(float)
    if len(resid) <= 1:
        return 0.1
    return float(max(np.std(resid, ddof=1), 1e-3))


def predict_residual_gaussian(trained: TrainedModel, frame: pd.DataFrame, sigma: float) -> pd.DataFrame:
    pred = _predict_pass(trained, frame, train_mode=False)
    pred["logv_mean"] = pred["logv_pred"]
    half = 1.959963984540054 * sigma
    pred["logv_lower"] = pred["logv_mean"] - half
    pred["logv_upper"] = pred["logv_mean"] + half
    return pred[["trajectory_id", "logv_target", "logv_mean", "logv_lower", "logv_upper", "physics_residual", "physics_residual_abs"]]


def predict_laplace_approx(trained: TrainedModel, train_frame: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """A practical Gaussian/Laplace-style approximation using residual variance."""
    sigma = estimate_residual_sigma(trained, train_frame)
    return predict_residual_gaussian(trained, frame, sigma=sigma)


def predict_hmc_placeholder(*_args, **_kwargs) -> pd.DataFrame:
    raise NotImplementedError("Run HMC only on the NLSTt-300 high-cost cohort, preferably via a separate script.")
