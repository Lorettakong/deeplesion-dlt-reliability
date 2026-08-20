from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class HMCPrior:
    mean: np.ndarray
    std: np.ndarray
    obs_sigma: float


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x) + 1e-5


def _gompertz_y(t: torch.Tensor, raw_alpha: torch.Tensor, log_k: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
    alpha = _softplus(raw_alpha)
    return log_k - (log_k - y0) * torch.exp(-alpha * t)


def _fit_single_gompertz(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    y0 = float(y[0])
    y_max = float(np.max(y))
    best = (float("inf"), 0.2, y_max + 0.5)
    logk_grid = np.linspace(y_max + 0.02, y_max + 2.0, 50)
    alpha_grid = np.linspace(0.02, 3.0, 70)
    for log_k in logk_grid:
        pred = log_k - (log_k - y0) * np.exp(-alpha_grid[:, None] * t[None, :])
        losses = np.mean((pred - y[None, :]) ** 2, axis=1)
        idx = int(np.argmin(losses))
        if float(losses[idx]) < best[0]:
            best = (float(losses[idx]), float(alpha_grid[idx]), float(log_k))
    loss, alpha, log_k = best
    raw_alpha = float(np.log(np.expm1(max(alpha, 1e-5))))
    return raw_alpha, log_k, y0, loss


def estimate_hmc_prior(train_df: pd.DataFrame) -> HMCPrior:
    params = []
    losses = []
    for _, row in train_df.iterrows():
        t = np.asarray(row["t_obs"] + [row["t_target"]], dtype=float)
        y = np.asarray(row["logv_obs"] + [row["logv_target"]], dtype=float)
        raw_alpha, log_k, y0, loss = _fit_single_gompertz(t, y)
        params.append([raw_alpha, log_k, y0])
        losses.append(loss)
    arr = np.asarray(params, dtype=float)
    mean = np.mean(arr, axis=0)
    std = np.clip(np.std(arr, axis=0, ddof=1), 0.15, None)
    obs_sigma = float(max(np.sqrt(np.mean(losses)), 0.06))
    return HMCPrior(mean=mean, std=std, obs_sigma=obs_sigma)


def _potential(theta: torch.Tensor, t_obs: torch.Tensor, y_obs: torch.Tensor, prior: HMCPrior) -> torch.Tensor:
    mean = torch.tensor(prior.mean, dtype=torch.float32)
    std = torch.tensor(prior.std, dtype=torch.float32)
    sigma = torch.tensor(prior.obs_sigma, dtype=torch.float32)
    raw_alpha, log_k, y0 = theta[0], theta[1], theta[2]
    pred = _gompertz_y(t_obs, raw_alpha, log_k, y0)
    nll = 0.5 * torch.sum(((y_obs - pred) / sigma) ** 2) + len(y_obs) * torch.log(sigma)
    prior_nll = 0.5 * torch.sum(((theta - mean) / std) ** 2) + torch.sum(torch.log(std))
    return nll + prior_nll


def _grad_potential(theta_np: np.ndarray, t_obs_np: np.ndarray, y_obs_np: np.ndarray, prior: HMCPrior) -> tuple[float, np.ndarray]:
    theta = torch.tensor(theta_np, dtype=torch.float32, requires_grad=True)
    t_obs = torch.tensor(t_obs_np, dtype=torch.float32)
    y_obs = torch.tensor(y_obs_np, dtype=torch.float32)
    u = _potential(theta, t_obs, y_obs, prior)
    u.backward()
    return float(u.detach().cpu().numpy()), theta.grad.detach().cpu().numpy().astype(float)


def hmc_sample_case(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    t_target: float,
    prior: HMCPrior,
    seed: int,
    num_samples: int = 250,
    burn_in: int = 100,
    step_size: float = 0.015,
    leapfrog_steps: int = 15,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    theta = prior.mean.copy()
    samples = []
    accepts = 0
    total = num_samples + burn_in
    current_u, current_grad = _grad_potential(theta, t_obs, y_obs, prior)
    for i in range(total):
        q = theta.copy()
        p = rng.normal(size=3)
        current_p = p.copy()
        u = current_u
        grad = current_grad
        p = p - 0.5 * step_size * grad
        for lf in range(leapfrog_steps):
            q = q + step_size * p
            u, grad = _grad_potential(q, t_obs, y_obs, prior)
            if lf != leapfrog_steps - 1:
                p = p - step_size * grad
        p = p - 0.5 * step_size * grad
        p = -p
        current_h = current_u + 0.5 * np.sum(current_p**2)
        proposed_h = u + 0.5 * np.sum(p**2)
        accept_prob = min(1.0, float(np.exp(np.clip(current_h - proposed_h, -50, 50))))
        if rng.uniform() < accept_prob:
            theta = q
            current_u = u
            current_grad = grad
            accepts += 1
        if i >= burn_in:
            samples.append(theta.copy())
    samples_np = np.asarray(samples)
    theta_t = torch.tensor(samples_np, dtype=torch.float32)
    tt = torch.tensor(float(t_target), dtype=torch.float32)
    pred = _gompertz_y(tt, theta_t[:, 0], theta_t[:, 1], theta_t[:, 2]).detach().cpu().numpy()
    return pred.astype(float), accepts / max(total, 1)


def predict_hmc_gompertz(
    train_df: pd.DataFrame,
    frame: pd.DataFrame,
    seed: int,
    num_samples: int = 250,
    burn_in: int = 100,
) -> pd.DataFrame:
    prior = estimate_hmc_prior(train_df)
    rows = []
    accept_rates = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        t_obs = np.asarray(row["t_obs"], dtype=float)
        y_obs = np.asarray(row["logv_obs"], dtype=float)
        draws, accept = hmc_sample_case(
            t_obs=t_obs,
            y_obs=y_obs,
            t_target=float(row["t_target"]),
            prior=prior,
            seed=seed + idx,
            num_samples=num_samples,
            burn_in=burn_in,
        )
        accept_rates.append(accept)
        rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "logv_target": float(row["logv_target"]),
                "logv_mean": float(np.mean(draws)),
                "logv_lower": float(np.quantile(draws, 0.025)),
                "logv_upper": float(np.quantile(draws, 0.975)),
                "hmc_accept_rate": float(accept),
            }
        )
    return pd.DataFrame(rows)

