from __future__ import annotations

import torch


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

