from __future__ import annotations

import numpy as np
import pandas as pd


def _gompertz_fit_predict(t_obs: np.ndarray, y_obs: np.ndarray, t_target: float) -> float:
    y0 = float(y_obs[0])
    y_max = float(np.max(y_obs))
    logk_grid = np.linspace(y_max + 0.02, y_max + 2.0, 80)
    alpha_grid = np.linspace(0.01, 3.0, 120)
    best_loss = float("inf")
    best_pred = float(y_obs[-1])
    for log_k in logk_grid:
        base = log_k - y0
        pred_obs_base = log_k - base * np.exp(-alpha_grid[:, None] * t_obs[None, :])
        losses = np.mean((pred_obs_base - y_obs[None, :]) ** 2, axis=1)
        idx = int(np.argmin(losses))
        if float(losses[idx]) < best_loss:
            alpha = float(alpha_grid[idx])
            best_loss = float(losses[idx])
            best_pred = float(log_k - base * np.exp(-alpha * t_target))
    return best_pred


def _logistic_fit_predict(t_obs: np.ndarray, y_obs: np.ndarray, t_target: float) -> float:
    v_obs = np.exp(y_obs)
    v0 = float(max(v_obs[0], 1e-8))
    vmax = float(np.max(v_obs))
    k_grid = np.linspace(max(vmax * 1.02, v0 * 1.05), max(vmax * 8.0, v0 * 2.0), 80)
    r_grid = np.linspace(0.01, 4.0, 140)
    best_loss = float("inf")
    best_pred = float(y_obs[-1])
    for k_cap in k_grid:
        a = k_cap / v0 - 1.0
        if a <= 0:
            continue
        pred_v = k_cap / (1.0 + a * np.exp(-r_grid[:, None] * t_obs[None, :]))
        pred_y = np.log(np.clip(pred_v, 1e-8, None))
        losses = np.mean((pred_y - y_obs[None, :]) ** 2, axis=1)
        idx = int(np.argmin(losses))
        if float(losses[idx]) < best_loss:
            r = float(r_grid[idx])
            best_loss = float(losses[idx])
            pred_v_target = k_cap / (1.0 + a * np.exp(-r * t_target))
            best_pred = float(np.log(max(pred_v_target, 1e-8)))
    return best_pred


def _log_linear_fit_predict(t_obs: np.ndarray, y_obs: np.ndarray, t_target: float) -> float:
    if len(t_obs) == 1:
        return float(y_obs[0])
    slope, intercept = np.polyfit(t_obs, y_obs, 1)
    return float(intercept + slope * t_target)


def population_growth_prior(dense_df: pd.DataFrame, train_ids: list[str]) -> float:
    slopes = []
    for _, group in dense_df[dense_df["trajectory_id"].isin(train_ids)].groupby("trajectory_id"):
        group = group.sort_values("t_rel")
        t = group["t_rel"].to_numpy(float)
        y = group["logV"].to_numpy(float)
        dt = t[-1] - t[0]
        if dt > 0:
            slopes.append((y[-1] - y[0]) / dt)
    return float(np.median(slopes)) if slopes else 0.0


def predict_density_mechanistic(
    dense_df: pd.DataFrame,
    test_ids: list[str],
    k: int,
    model: str,
    prior_slope: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for tid, group in dense_df[dense_df["trajectory_id"].isin(test_ids)].groupby("trajectory_id"):
        group = group.sort_values("t_rel")
        if len(group) <= k:
            continue
        obs = group.iloc[:k]
        target = group.iloc[-1]
        t_obs = obs["t_rel"].to_numpy(float)
        y_obs = obs["logV"].to_numpy(float)
        t_target = float(target["t_rel"])
        if k == 1:
            y_pred = float(y_obs[0] + prior_slope * (t_target - t_obs[0]))
        elif model == "gompertz":
            y_pred = _gompertz_fit_predict(t_obs, y_obs, t_target)
        elif model == "logistic":
            y_pred = _logistic_fit_predict(t_obs, y_obs, t_target)
        elif model == "log_linear":
            y_pred = _log_linear_fit_predict(t_obs, y_obs, t_target)
        else:
            raise ValueError(f"Unknown mechanistic model: {model}")
        rows.append(
            {
                "trajectory_id": str(tid),
                "logv_target": float(target["logV"]),
                "logv_mean": y_pred,
                "logv_lower": y_pred,
                "logv_upper": y_pred,
                "generator": group["generator"].iloc[0],
                "total_nodes": int(group["total_nodes"].iloc[0]),
                "k": k,
                "fit_model": model,
            }
        )
    return pd.DataFrame(rows)


def predict_density_staged_mechanistic(
    dense_df: pd.DataFrame,
    test_ids: list[str],
    k: int,
    prior_slope: float = 0.0,
) -> pd.DataFrame:
    """Use identifiable model complexity for each observation count.

    k=1: persistence baseline because individual growth rate is not identifiable.
    k=2: log-linear extrapolation because two points identify a slope.
    k>=3: Gompertz form because nonlinear curvature becomes partially identifiable.
    """
    rows = []
    for tid, group in dense_df[dense_df["trajectory_id"].isin(test_ids)].groupby("trajectory_id"):
        group = group.sort_values("t_rel")
        if len(group) <= k:
            continue
        obs = group.iloc[:k]
        target = group.iloc[-1]
        t_obs = obs["t_rel"].to_numpy(float)
        y_obs = obs["logV"].to_numpy(float)
        t_target = float(target["t_rel"])
        if k == 1:
            y_pred = float(y_obs[0])
            model = "persistence"
        elif k == 2:
            y_pred = _log_linear_fit_predict(t_obs, y_obs, t_target)
            model = "log_linear"
        else:
            y_pred = _gompertz_fit_predict(t_obs, y_obs, t_target)
            model = "gompertz"
        rows.append(
            {
                "trajectory_id": str(tid),
                "logv_target": float(target["logV"]),
                "logv_mean": y_pred,
                "logv_lower": y_pred,
                "logv_upper": y_pred,
                "generator": group["generator"].iloc[0],
                "total_nodes": int(group["total_nodes"].iloc[0]),
                "k": k,
                "fit_model": model,
                "prior_slope": prior_slope,
            }
        )
    return pd.DataFrame(rows)
