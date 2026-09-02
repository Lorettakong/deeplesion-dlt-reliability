from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_mean: np.ndarray) -> dict[str, float]:
    err = y_mean - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / np.clip(np.abs(y_true), 1e-8, None))),
    }


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    covered = (y_true >= lower) & (y_true <= upper)
    return {
        "picp": float(np.mean(covered)),
        "covered_count": int(np.sum(covered)),
        "evaluation_count": int(len(covered)),
        "mpiw": float(np.mean(upper - lower)),
    }


def interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Mean central prediction-interval score (lower is better)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one.")
    score = (
        (upper - lower)
        + (2.0 / alpha) * (lower - y_true) * (y_true < lower)
        + (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    )
    return float(np.mean(score))


def one_level_weighted_interval_score(
    y_true: np.ndarray,
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """WIS using the predictive median and one central interval level."""
    is_value = interval_score(y_true, lower, upper, alpha=alpha)
    numerator = 0.5 * float(np.mean(np.abs(y_true - median))) + (alpha / 2.0) * is_value
    return numerator / (0.5 + alpha / 2.0)


def gaussian_nll(y_true: np.ndarray, y_mean: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    sigma = np.clip((upper - lower) / (2.0 * 1.959963984540054), 1e-6, None)
    nll = 0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * ((y_true - y_mean) / sigma) ** 2
    return float(np.mean(nll))


def interval_ece(y_true: np.ndarray, y_mean: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    sigma = np.clip((upper - lower) / (2.0 * 1.959963984540054), 1e-6, None)
    z_values = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.959963984540054])
    nominal = np.array([0.1974, 0.3829, 0.5467, 0.6827, 0.7887, 0.8664, 0.9199, 0.9500])
    empirical = []
    abs_err = np.abs(y_true - y_mean)
    for z in z_values:
        empirical.append(float(np.mean(abs_err <= z * sigma)))
    return float(np.mean(np.abs(np.array(empirical) - nominal)))


def summarize_predictions(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["logv_target"].to_numpy(float)
    y_mean = df["logv_mean"].to_numpy(float)
    out = regression_metrics(y_true, y_mean)
    if {"logv_lower", "logv_upper"}.issubset(df.columns):
        lower = df["logv_lower"].to_numpy(float)
        upper = df["logv_upper"].to_numpy(float)
        out.update(interval_metrics(y_true, lower, upper))
        alpha = (
            float(df["interval_alpha"].iloc[0])
            if "interval_alpha" in df.columns
            else 0.05
        )
        out["interval_alpha"] = alpha
        out["interval_score"] = interval_score(y_true, lower, upper, alpha=alpha)
        out["wis_1level"] = one_level_weighted_interval_score(
            y_true, y_mean, lower, upper, alpha=alpha
        )
        is_conformal = (
            "interval_type" in df.columns
            and df["interval_type"].astype(str).str.startswith("conformal").all()
        )
        if "predictive_log_prob" in df.columns and not is_conformal:
            out["nll"] = float(-np.mean(df["predictive_log_prob"].to_numpy(float)))
            out["nll_definition"] = "predictive_density"
        elif not is_conformal:
            out["nll"] = gaussian_nll(y_true, y_mean, lower, upper)
            out["nll_definition"] = "working_gaussian_from_interval"
        else:
            out["nll"] = float("nan")
            out["nll_definition"] = "not_defined_for_conformal_interval"
        out["ece"] = float("nan") if is_conformal else interval_ece(y_true, y_mean, lower, upper)
    if "physics_residual_abs" in df.columns:
        out["physics_residual_abs"] = float(np.mean(df["physics_residual_abs"].to_numpy(float)))
    elif "physics_residual" in df.columns:
        out["physics_residual_abs"] = float(np.mean(np.abs(df["physics_residual"].to_numpy(float))))
    return out


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int, float]:
    """Finite-sample split-conformal quantile.

    Returns the q score, one-based rank, and rank/n quantile level. For n
    calibration samples, the selected rank is ceil((n + 1) * (1 - alpha)),
    clipped to n. This makes the order statistic explicit and avoids
    interpolation-dependent behavior in np.quantile.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration scores are empty.")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    sorted_scores = np.sort(scores)
    return float(sorted_scores[rank - 1]), rank, rank / n


def residual_calibrate_intervals(
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, float]:
    """Calibrate intervals using validation absolute residuals around the predictive mean."""
    scores = np.abs(val_pred["logv_target"].to_numpy(float) - val_pred["logv_mean"].to_numpy(float))
    q, _, _ = split_conformal_quantile(scores, alpha)
    out = test_pred.copy()
    out["logv_lower_raw"] = out.get("logv_lower", out["logv_mean"])
    out["logv_upper_raw"] = out.get("logv_upper", out["logv_mean"])
    out["logv_lower"] = out["logv_mean"] - q
    out["logv_upper"] = out["logv_mean"] + q
    out["interval_type"] = "conformal_absolute_residual"
    out["interval_alpha"] = alpha
    out["predictive_log_prob"] = np.nan
    return out, q


def conformalize_existing_intervals(
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, float]:
    """Scale-calibrate existing intervals using split conformal normalized residuals."""
    required = {"logv_target", "logv_mean", "logv_lower", "logv_upper"}
    if not required.issubset(val_pred.columns) or not required.issubset(test_pred.columns):
        return residual_calibrate_intervals(val_pred, test_pred, alpha=alpha)
    y = val_pred["logv_target"].to_numpy(float)
    mean = val_pred["logv_mean"].to_numpy(float)
    sigma = np.clip((val_pred["logv_upper"].to_numpy(float) - val_pred["logv_lower"].to_numpy(float)) / (2.0 * 1.959963984540054), 1e-6, None)
    scores = np.abs(y - mean) / sigma
    q, _, _ = split_conformal_quantile(scores, alpha)
    out = test_pred.copy()
    out["logv_lower_raw"] = out["logv_lower"]
    out["logv_upper_raw"] = out["logv_upper"]
    test_sigma = np.clip((out["logv_upper"].to_numpy(float) - out["logv_lower"].to_numpy(float)) / (2.0 * 1.959963984540054), 1e-6, None)
    half_width = q * test_sigma
    out["logv_lower"] = out["logv_mean"] - half_width
    out["logv_upper"] = out["logv_mean"] + half_width
    out["interval_type"] = "conformal_normalized_residual"
    out["interval_alpha"] = alpha
    out["predictive_log_prob"] = np.nan
    return out, q


def _patient_aggregate_scores(
    calibration_pred: pd.DataFrame,
    scores: np.ndarray,
    patient_col: str = "patient_id",
) -> np.ndarray:
    """Return one conservative nonconformity score per calibration patient."""
    if patient_col not in calibration_pred.columns:
        raise ValueError(
            f"Patient-level conformal calibration requires {patient_col!r}."
        )
    scored = pd.DataFrame(
        {patient_col: calibration_pred[patient_col].astype(str), "score": scores}
    )
    return scored.groupby(patient_col, sort=True)["score"].max().to_numpy(float)


def patient_level_residual_calibrate_intervals(
    calibration_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    alpha: float = 0.05,
    patient_col: str = "patient_id",
) -> tuple[pd.DataFrame, float, dict[str, float | int | str]]:
    """Patient-cluster split conformal using one maximum residual per patient."""
    trajectory_scores = np.abs(
        calibration_pred["logv_target"].to_numpy(float)
        - calibration_pred["logv_mean"].to_numpy(float)
    )
    patient_scores = _patient_aggregate_scores(
        calibration_pred, trajectory_scores, patient_col=patient_col
    )
    q, rank, level = split_conformal_quantile(patient_scores, alpha)
    out = test_pred.copy()
    out["logv_lower_raw"] = out.get("logv_lower", out["logv_mean"])
    out["logv_upper_raw"] = out.get("logv_upper", out["logv_mean"])
    out["logv_lower"] = out["logv_mean"] - q
    out["logv_upper"] = out["logv_mean"] + q
    out["interval_type"] = "conformal_patient_cluster_absolute"
    out["interval_alpha"] = alpha
    out["predictive_log_prob"] = np.nan
    metadata = {
        "calibration_unit": "patient",
        "patient_score_aggregation": "maximum",
        "n_calibration_patients": int(len(patient_scores)),
        "conformal_rank": int(rank),
        "conformal_level": float(level),
    }
    return out, q, metadata


def patient_level_conformalize_existing_intervals(
    calibration_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    alpha: float = 0.05,
    patient_col: str = "patient_id",
) -> tuple[pd.DataFrame, float, dict[str, float | int | str]]:
    """Patient-cluster normalized split conformal for predictive scales."""
    required = {"logv_target", "logv_mean", "logv_lower", "logv_upper", patient_col}
    if not required.issubset(calibration_pred.columns) or not required.issubset(test_pred.columns):
        return patient_level_residual_calibrate_intervals(
            calibration_pred, test_pred, alpha=alpha, patient_col=patient_col
        )
    cal_sigma = np.clip(
        (
            calibration_pred["logv_upper"].to_numpy(float)
            - calibration_pred["logv_lower"].to_numpy(float)
        )
        / (2.0 * 1.959963984540054),
        1e-6,
        None,
    )
    trajectory_scores = np.abs(
        calibration_pred["logv_target"].to_numpy(float)
        - calibration_pred["logv_mean"].to_numpy(float)
    ) / cal_sigma
    patient_scores = _patient_aggregate_scores(
        calibration_pred, trajectory_scores, patient_col=patient_col
    )
    q, rank, level = split_conformal_quantile(patient_scores, alpha)
    out = test_pred.copy()
    out["logv_lower_raw"] = out["logv_lower"]
    out["logv_upper_raw"] = out["logv_upper"]
    test_sigma = np.clip(
        (out["logv_upper"].to_numpy(float) - out["logv_lower"].to_numpy(float))
        / (2.0 * 1.959963984540054),
        1e-6,
        None,
    )
    half_width = q * test_sigma
    out["logv_lower"] = out["logv_mean"] - half_width
    out["logv_upper"] = out["logv_mean"] + half_width
    out["interval_type"] = "conformal_patient_cluster_normalized"
    out["interval_alpha"] = alpha
    out["predictive_log_prob"] = np.nan
    metadata = {
        "calibration_unit": "patient",
        "patient_score_aggregation": "maximum",
        "n_calibration_patients": int(len(patient_scores)),
        "conformal_rank": int(rank),
        "conformal_level": float(level),
    }
    return out, q, metadata
