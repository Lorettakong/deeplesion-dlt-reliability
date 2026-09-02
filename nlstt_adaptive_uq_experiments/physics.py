from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares


@dataclass(frozen=True)
class FixedGompertzReference:
    """Training-only Gompertz reference used by the residual."""

    alpha: float
    log_k: float
    y0: float
    n_trajectories: int
    fit_rmse: float


def estimate_fixed_gompertz_reference(
    long_df: pd.DataFrame,
    train_ids: list[str],
) -> FixedGompertzReference:
    """Fit one population reference to the mean training trajectory.

    Parameters are estimated once from training trajectories and must then be
    frozen for model fitting, validation, and testing.  No model prediction or
    validation/test target is used in this fit.
    """
    required = {"trajectory_id", "t_rel", "logV"}
    missing = required.difference(long_df.columns)
    if missing:
        raise ValueError(f"Cannot estimate Gompertz reference; missing columns: {sorted(missing)}")
    train_set = {str(x) for x in train_ids}
    frame = long_df[long_df["trajectory_id"].astype(str).isin(train_set)].copy()
    if frame.empty:
        raise ValueError("No training trajectories were supplied for Gompertz reference fitting.")
    mean_trajectory = frame.groupby("t_rel", as_index=False)["logV"].mean().sort_values("t_rel")
    t = mean_trajectory["t_rel"].to_numpy(float)
    y = mean_trajectory["logV"].to_numpy(float)
    if len(t) < 3:
        raise ValueError("At least three mean-trajectory time points are required.")
    t0 = float(t[0])
    y0 = float(y[0])

    def residual(theta: np.ndarray) -> np.ndarray:
        alpha = float(np.exp(theta[0]))
        log_k = float(theta[1])
        pred = log_k - (log_k - y0) * np.exp(-alpha * (t - t0))
        return pred - y

    slope = float((y[-1] - y[0]) / max(t[-1] - t[0], 1e-6))
    initial_log_k = float(y[-1] + np.sign(slope if slope != 0 else 1.0) * max(abs(y[-1] - y0), 0.25))
    fit = least_squares(
        residual,
        x0=np.asarray([np.log(0.25), initial_log_k], dtype=float),
        bounds=(np.asarray([np.log(1e-4), -20.0]), np.asarray([np.log(10.0), 20.0])),
    )
    alpha = float(np.exp(fit.x[0]))
    log_k = float(fit.x[1])
    fit_rmse = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
    return FixedGompertzReference(
        alpha=alpha,
        log_k=log_k,
        y0=y0,
        n_trajectories=int(frame["trajectory_id"].nunique()),
        fit_rmse=fit_rmse,
    )


def gompertz_rhs(log_v: torch.Tensor, alpha: torch.Tensor, log_k: torch.Tensor) -> torch.Tensor:
    """d log(V) / dt for Gompertz growth: y' = alpha * (logK - y)."""
    return alpha * (log_k - log_v)


def logistic_rhs(v: torch.Tensor, r: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """dV/dt for logistic growth."""
    return r * v * (1.0 - v / torch.clamp(k, min=1e-6))


def finite_difference_residual(t: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor, log_k: torch.Tensor) -> torch.Tensor:
    """Approximate Gompertz physics residual on a predicted trajectory."""
    dt = torch.clamp(t[:, 1:] - t[:, :-1], min=1e-6)
    dy = y[:, 1:] - y[:, :-1]
    y_mid = 0.5 * (y[:, 1:] + y[:, :-1])
    rhs = gompertz_rhs(y_mid, alpha[:, None], log_k[:, None])
    return dy / dt - rhs
