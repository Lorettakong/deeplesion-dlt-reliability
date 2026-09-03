from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import DataLoader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import pad_batch
from .train import HETEROSCEDASTIC_METHODS, MIXTURE_PHYSICS_NAMES, TrajectoryDataset, TrainedModel

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def _gp_features(frame: pd.DataFrame) -> np.ndarray:
    """Fixed-length features for one recent-history task (one GP is fit per m)."""
    rows = []
    for record in frame.to_dict("records"):
        t = np.asarray(record["t_obs"], dtype=float)
        y = np.asarray(record["logv_obs"], dtype=float)
        if len(t) != len(y):
            raise ValueError("t_obs and logv_obs must have equal length.")
        # Relative times improve conditioning while retaining the actual forecast gap.
        t0 = float(t[0])
        rows.append(np.r_[t - t0, y, float(record["t_target"]) - t0])
    return np.asarray(rows, dtype=float)


def fit_gaussian_process(train_frame: pd.DataFrame, seed: int = 0) -> Pipeline:
    """Cohort-level GP on standardized history features for Experiment 2."""
    x = _gp_features(train_frame)
    y = train_frame["logv_target"].to_numpy(float)
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(x.shape[1]), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-6, 1e1))
    )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "gp",
                GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    n_restarts_optimizer=3,
                    random_state=seed,
                ),
            ),
        ]
    ).fit(x, y)


def predict_gaussian_process(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    x = model.named_steps["scale"].transform(_gp_features(frame))
    mean, std = model.named_steps["gp"].predict(x, return_std=True)
    std = np.clip(np.asarray(std, dtype=float), 1e-6, None)
    target = frame["logv_target"].to_numpy(float)
    out = pd.DataFrame(
        {
            "trajectory_id": frame["trajectory_id"].astype(str).to_numpy(),
            "patient_id": frame["patient_id"].astype(str).to_numpy(),
            "logv_target": target,
            "logv_mean": mean,
            "logv_lower": mean - 1.959963984540054 * std,
            "logv_upper": mean + 1.959963984540054 * std,
            "aleatoric_std": np.nan,
            "epistemic_std": np.nan,
            "total_std": std,
            "uncertainty_definition": "gp_posterior_predictive",
            "interval_type": "gp_posterior_predictive",
            "interval_alpha": 0.05,
            "physics_residual": np.nan,
            "physics_residual_abs": np.nan,
        }
    )
    var = std**2
    out["predictive_log_prob"] = -0.5 * (
        np.log(2.0 * np.pi * var) + (target - mean) ** 2 / var
    )
    return out


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
                reference = trained.gompertz_reference
                if reference is None:
                    physics_residual = float("nan")
                    alpha = float("nan")
                    log_k = float("nan")
                else:
                    alpha = float(reference.alpha)
                    log_k = float(reference.log_k)
                    rhs = alpha * (log_k - 0.5 * (last_y + pred_y))
                    physics_residual = dy_dt - rhs
                row = {
                    "trajectory_id": tid,
                    "patient_id": batch["patient_ids"][i],
                    "logv_target": float(batch["y_target"][i]),
                    "logv_pred": pred_y,
                    "physics_residual": float(physics_residual),
                    "physics_residual_abs": float(abs(physics_residual)),
                    "gompertz_alpha_fixed": alpha,
                    "gompertz_log_k_fixed": log_k,
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
    draws = [
        _predict_pass(trained, frame, train_mode=True).rename(
            columns={"logv_pred": f"draw_{i}", "physics_residual": f"residual_{i}"}
        )
        for i in range(samples)
    ]
    merged = draws[0][["trajectory_id", "patient_id", "logv_target"]].copy()
    for i, draw in enumerate(draws):
        merged = merged.merge(draw[["trajectory_id", f"draw_{i}", f"residual_{i}"]], on="trajectory_id")
        if "log_var" in draw.columns:
            merged = merged.merge(
                draw[["trajectory_id", "log_var"]].rename(columns={"log_var": f"log_var_{i}"}),
                on="trajectory_id",
            )
    draw_cols = [c for c in merged.columns if c.startswith("draw_")]
    residual_cols = [c for c in merged.columns if c.startswith("residual_")]
    vals = merged[draw_cols].to_numpy(float)
    merged["logv_mean"] = vals.mean(axis=1)
    epistemic_var = vals.var(axis=1)
    log_var_cols = [c for c in merged.columns if c.startswith("log_var_")]
    if log_var_cols:
        component_log_var = np.clip(merged[log_var_cols].to_numpy(float), -8.0, 4.0)
        component_var = np.exp(component_log_var)
        aleatoric_var = component_var.mean(axis=1)
        uncertainty_definition = "aleatoric_plus_epistemic"
    else:
        component_log_var = None
        aleatoric_var = np.zeros(len(merged), dtype=float)
        uncertainty_definition = "epistemic_only"
    total_var = np.clip(aleatoric_var + epistemic_var, 1e-8, None)
    total_std = np.sqrt(total_var)
    merged["aleatoric_std"] = np.sqrt(np.clip(aleatoric_var, 0.0, None))
    merged["epistemic_std"] = np.sqrt(np.clip(epistemic_var, 0.0, None))
    merged["total_std"] = total_std
    merged["logv_lower"] = merged["logv_mean"] - 1.959963984540054 * total_std
    merged["logv_upper"] = merged["logv_mean"] + 1.959963984540054 * total_std
    if component_log_var is not None:
        y = merged["logv_target"].to_numpy(float)[:, None]
        log_density = -0.5 * (
            np.log(2.0 * np.pi)
            + component_log_var
            + (y - vals) ** 2 / np.exp(component_log_var)
        )
        merged["predictive_log_prob"] = np.logaddexp.reduce(log_density, axis=1) - np.log(samples)
    else:
        merged["predictive_log_prob"] = -0.5 * (
            np.log(2.0 * np.pi * total_var)
            + (merged["logv_target"].to_numpy(float) - merged["logv_mean"].to_numpy(float)) ** 2 / total_var
        )
    merged["uncertainty_definition"] = uncertainty_definition
    merged["interval_type"] = "moment_matched_predictive"
    merged["interval_alpha"] = 0.05
    merged["physics_residual"] = merged[residual_cols].mean(axis=1)
    merged["physics_residual_abs"] = merged["physics_residual"].abs()
    return merged[[
        "trajectory_id", "patient_id", "logv_target", "logv_mean", "logv_lower", "logv_upper",
        "aleatoric_std", "epistemic_std", "total_std", "predictive_log_prob",
        "uncertainty_definition", "interval_type", "interval_alpha", "physics_residual", "physics_residual_abs",
    ]]


def predict_ensemble(models: list[TrainedModel], frame: pd.DataFrame) -> pd.DataFrame:
    draws = []
    for i, model in enumerate(models):
        draw = _predict_pass(model, frame).rename(
            columns={"logv_pred": f"draw_{i}", "physics_residual": f"residual_{i}"}
        )
        if "log_var" in draw.columns:
            draw = draw.rename(columns={"log_var": f"log_var_{i}"})
        draws.append(draw)
    merged = draws[0][["trajectory_id", "patient_id", "logv_target"]].copy()
    for i, draw in enumerate(draws):
        merged = merged.merge(draw[["trajectory_id", f"draw_{i}", f"residual_{i}"]], on="trajectory_id")
        if f"log_var_{i}" in draw.columns:
            merged = merged.merge(draw[["trajectory_id", f"log_var_{i}"]], on="trajectory_id")
    draw_cols = [c for c in merged.columns if c.startswith("draw_")]
    residual_cols = [c for c in merged.columns if c.startswith("residual_")]
    vals = merged[draw_cols].to_numpy(float)
    merged["logv_mean"] = vals.mean(axis=1)
    merged["physics_residual"] = merged[residual_cols].mean(axis=1)
    merged["physics_residual_abs"] = merged["physics_residual"].abs()
    log_var_cols = [c for c in merged.columns if c.startswith("log_var_")]
    if log_var_cols:
        component_log_var = np.clip(merged[log_var_cols].to_numpy(float), -8.0, 4.0)
        component_var = np.exp(component_log_var)
        aleatoric_var = component_var.mean(axis=1)
        epistemic_var = vals.var(axis=1)
        total_std = np.sqrt(np.clip(aleatoric_var + epistemic_var, 1e-8, None))
        merged["logv_lower"] = merged["logv_mean"] - 1.959963984540054 * total_std
        merged["logv_upper"] = merged["logv_mean"] + 1.959963984540054 * total_std
        merged["aleatoric_std"] = np.sqrt(np.clip(aleatoric_var, 0.0, None))
        merged["epistemic_std"] = np.sqrt(np.clip(epistemic_var, 0.0, None))
        merged["total_std"] = total_std
        y = merged["logv_target"].to_numpy(float)[:, None]
        log_density = -0.5 * (
            np.log(2.0 * np.pi)
            + component_log_var
            + (y - vals) ** 2 / component_var
        )
        merged["predictive_log_prob"] = np.logaddexp.reduce(log_density, axis=1) - np.log(len(draws))
        merged["uncertainty_definition"] = "aleatoric_plus_epistemic"
        merged["interval_type"] = "moment_matched_predictive"
        merged["interval_alpha"] = 0.05
        keep = [
            "trajectory_id",
            "patient_id",
            "logv_target",
            "logv_mean",
            "logv_lower",
            "logv_upper",
            "physics_residual",
            "physics_residual_abs",
            "aleatoric_std",
            "epistemic_std",
            "total_std",
            "predictive_log_prob",
            "uncertainty_definition",
            "interval_type",
            "interval_alpha",
        ]
    else:
        merged["logv_lower"] = np.quantile(vals, 0.025, axis=1)
        merged["logv_upper"] = np.quantile(vals, 0.975, axis=1)
        merged["aleatoric_std"] = 0.0
        merged["epistemic_std"] = vals.std(axis=1)
        merged["total_std"] = np.clip(merged["epistemic_std"], 1e-4, None)
        total_var = merged["total_std"].to_numpy(float) ** 2
        merged["predictive_log_prob"] = -0.5 * (
            np.log(2.0 * np.pi * total_var)
            + (merged["logv_target"].to_numpy(float) - merged["logv_mean"].to_numpy(float)) ** 2 / total_var
        )
        merged["uncertainty_definition"] = "epistemic_only"
        merged["interval_type"] = "empirical_quantile"
        merged["interval_alpha"] = 0.05
        keep = [
            "trajectory_id", "patient_id", "logv_target", "logv_mean", "logv_lower", "logv_upper",
            "physics_residual", "physics_residual_abs", "aleatoric_std", "epistemic_std", "total_std",
            "predictive_log_prob", "uncertainty_definition", "interval_type",
            "interval_alpha",
        ]
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
    pred["aleatoric_std"] = sigma
    pred["epistemic_std"] = 0.0
    pred["total_std"] = sigma
    var = max(sigma**2, 1e-8)
    pred["predictive_log_prob"] = -0.5 * (
        np.log(2.0 * np.pi * var) + (pred["logv_target"] - pred["logv_mean"]) ** 2 / var
    )
    pred["uncertainty_definition"] = "residual_aleatoric"
    pred["interval_type"] = "predictive_gaussian"
    pred["interval_alpha"] = 0.05
    return pred[[
        "trajectory_id", "patient_id", "logv_target", "logv_mean", "logv_lower", "logv_upper",
        "aleatoric_std", "epistemic_std", "total_std", "predictive_log_prob",
        "uncertainty_definition", "interval_type", "interval_alpha", "physics_residual", "physics_residual_abs",
    ]]


def predict_residual_scale(trained: TrainedModel, train_frame: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Gaussian predictive intervals using residual scale from development data."""
    sigma = estimate_residual_sigma(trained, train_frame)
    return predict_residual_gaussian(trained, frame, sigma=sigma)


def predict_laplace_approx(trained: TrainedModel, train_frame: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for released result filenames."""
    return predict_residual_scale(trained, train_frame, frame)
