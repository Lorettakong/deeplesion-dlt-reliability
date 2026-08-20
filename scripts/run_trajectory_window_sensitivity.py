from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlstt_adaptive_uq_experiments.mechanistic_baselines import (
    predict_density_staged_mechanistic,
)
from nlstt_adaptive_uq_experiments.config import ExperimentConfig
from nlstt_adaptive_uq_experiments.data import load_deeplesion_len5_long, split_trajectory_ids_by_patient


ROOT = Path(__file__).resolve().parents[1]
RAW_TRAJECTORIES = ROOT / "outputs_deeplesion_longitudinal" / "deeplesion_trajectories_len5.csv"
OUT_DIR = ROOT / "outputs_nlstt_adaptive_uq_paper" / "trajectory_window_sensitivity"


def _load_main_trajectory_splits() -> dict[str, set[str]]:
    """Recreate the patient-level split used by Experiment 2.

    Some persisted cohort CSVs were produced by earlier split runs. Recomputing
    the split here keeps this sensitivity analysis aligned with the current
    validation-selected Experiment 1/2 outputs.
    """
    cfg = ExperimentConfig()
    cohort = load_deeplesion_len5_long(relative_log=True)
    ids = sorted(cohort["trajectory_id"].astype(str).unique().tolist())
    split = split_trajectory_ids_by_patient(
        cohort,
        ids,
        seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )
    return {
        "train": set(split.train),
        "valid": set(split.val),
        "test": set(split.test),
    }


def _prepare_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_TRAJECTORIES)
    required = {
        "trajectory_id",
        "followup_index",
        "patient_id",
        "long_axis_mm",
        "short_axis_mm",
        "volume_proxy_mm3",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Trajectory file is missing columns: {sorted(missing)}")
    df = df.copy()
    df["trajectory_id"] = df["trajectory_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str).str.lstrip("0").replace("", "0")
    df["followup_index"] = df["followup_index"].astype(int)
    df = df.sort_values(["trajectory_id", "followup_index"])
    volume_cm3 = df["volume_proxy_mm3"].astype(float) / 1000.0
    fallback_cm3 = (
        math.pi
        / 6.0
        * df["long_axis_mm"].astype(float)
        * (df["short_axis_mm"].astype(float) ** 2)
        / 1000.0
    )
    df["V_obs_cm3_raw"] = np.where(np.isfinite(volume_cm3), volume_cm3, fallback_cm3)
    df["V_obs_cm3_raw"] = np.clip(df["V_obs_cm3_raw"], 1e-8, None)
    df["raw_logV"] = np.log(df["V_obs_cm3_raw"])
    return df


def _take_window(group: pd.DataFrame, indices: list[int], setting: str, window_idx: int) -> pd.DataFrame:
    sub = group.iloc[indices].copy()
    original_tid = str(sub["trajectory_id"].iloc[0])
    sub["original_trajectory_id"] = original_tid
    sub["trajectory_id"] = f"{original_tid}__{setting}_{window_idx:03d}"
    sub["window_setting"] = setting
    sub["window_index"] = window_idx
    sub["node_index"] = np.arange(len(sub), dtype=int)
    sub["t_rel"] = sub["node_index"].astype(float)
    sub["total_nodes"] = 5
    sub["generator"] = setting
    baseline = float(sub["raw_logV"].iloc[0])
    sub["logV_baseline"] = baseline
    sub["logV"] = sub["raw_logV"] - baseline
    sub["V_obs_cm3"] = np.exp(sub["logV"])
    rel_change = (float(sub["V_obs_cm3_raw"].iloc[-1]) - float(sub["V_obs_cm3_raw"].iloc[0])) / max(
        abs(float(sub["V_obs_cm3_raw"].iloc[0])), 1e-8
    )
    if rel_change < -0.20:
        growth_class = "decreasing"
    elif rel_change < 0.20:
        growth_class = "stable"
    elif rel_change < 1.00:
        growth_class = "slow_growth"
    else:
        growth_class = "rapid_growth"
    sub["growth_class"] = growth_class
    return sub


def build_windows(df: pd.DataFrame, setting: str, seed: int = 20260707) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, group in df.groupby("trajectory_id", sort=True):
        group = group.sort_values("followup_index").reset_index(drop=True)
        n = len(group)
        if n < 5:
            continue
        if setting == "first5":
            windows = [list(range(5))]
        elif setting == "last5":
            windows = [list(range(n - 5, n))]
        elif setting == "sliding5":
            windows = [list(range(start, start + 5)) for start in range(n - 4)]
        elif setting == "random5":
            if n == 5:
                choice = np.arange(5)
            else:
                choice = np.sort(rng.choice(n, size=5, replace=False))
            windows = [choice.astype(int).tolist()]
        else:
            raise ValueError(f"Unknown window setting: {setting}")
        for window_idx, indices in enumerate(windows):
            rows.append(_take_window(group, indices, setting, window_idx))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _bootstrap_ci(values: np.ndarray, seed: int = 7, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot.append(float(np.sqrt(np.mean(sample**2))))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def evaluate_setting(windows: pd.DataFrame, trajectory_splits: dict[str, set[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = windows.copy()
    windows["original_trajectory_id"] = windows["original_trajectory_id"].astype(str)
    split_by_tid = (
        windows.groupby("trajectory_id", as_index=False)
        .agg(original_trajectory_id=("original_trajectory_id", "first"), window_setting=("window_setting", "first"))
    )
    split_by_tid["split"] = "unused"
    for split, trajectory_ids in trajectory_splits.items():
        mask = split_by_tid["original_trajectory_id"].isin(trajectory_ids)
        split_by_tid.loc[mask, "split"] = split

    test_ids = split_by_tid.loc[split_by_tid["split"] == "test", "trajectory_id"].astype(str).tolist()
    metrics = []
    predictions = []
    for m in [1, 2, 3, 4]:
        pred = predict_density_staged_mechanistic(windows, test_ids, k=m)
        if pred.empty:
            continue
        pred["error"] = pred["logv_mean"] - pred["logv_target"]
        predictions.append(pred)
        rmse_lo, rmse_hi = _bootstrap_ci(pred["error"].to_numpy(float), seed=100 + m)
        metrics.append(
            {
                "window_setting": str(windows["window_setting"].iloc[0]),
                "m": m,
                "n_test_windows": int(len(pred)),
                "rmse": _rmse(pred["logv_target"].to_numpy(float), pred["logv_mean"].to_numpy(float)),
                "rmse_ci_low": rmse_lo,
                "rmse_ci_high": rmse_hi,
                "mae": _mae(pred["logv_target"].to_numpy(float), pred["logv_mean"].to_numpy(float)),
                "fit_rule": pred["fit_model"].iloc[0],
            }
        )
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return pd.DataFrame(metrics), pred_df


def plot_metrics(metrics: pd.DataFrame) -> None:
    order = ["first5", "last5", "sliding5", "random5"]
    labels = {
        "first5": "First-5",
        "last5": "Last-5",
        "sliding5": "Sliding-5",
        "random5": "Random-5",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for setting in order:
        sub = metrics[metrics["window_setting"] == setting].sort_values("m")
        if sub.empty:
            continue
        ax.plot(sub["m"], sub["rmse"], marker="o", linewidth=2, label=labels[setting])
        ax.fill_between(sub["m"], sub["rmse_ci_low"], sub["rmse_ci_high"], alpha=0.12)
    ax.set_xlabel("Observed visits used for prediction (m)")
    ax.set_ylabel("RMSE of relative log-volume")
    ax.set_title("Trajectory selection sensitivity")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_trajectory_window_sensitivity_rmse.png", dpi=220)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: f"{x:.4f}")
    headers = [str(c) for c in text.columns]
    rows = [[str(value) for value in row] for row in text.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = _prepare_raw()
    trajectory_splits = _load_main_trajectory_splits()

    all_metrics = []
    all_predictions = []
    window_summaries = []
    for setting in ["first5", "last5", "sliding5", "random5"]:
        windows = build_windows(raw, setting=setting)
        windows.to_csv(OUT_DIR / f"windows_{setting}.csv", index=False)
        metrics, predictions = evaluate_setting(windows, trajectory_splits)
        all_metrics.append(metrics)
        if not predictions.empty:
            predictions["window_setting"] = setting
            all_predictions.append(predictions)
        summary = (
            windows.groupby("trajectory_id", as_index=False)
            .agg(patient_id=("patient_id", "first"), original_trajectory_id=("original_trajectory_id", "first"))
        )
        window_summaries.append(
            {
                "window_setting": setting,
                "n_windows": int(len(summary)),
                "n_original_trajectories": int(summary["original_trajectory_id"].nunique()),
                "n_patients": int(summary["patient_id"].nunique()),
                "n_test_windows": int(summary[summary["original_trajectory_id"].isin(trajectory_splits["test"])].shape[0]),
            }
        )

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    window_summary_df = pd.DataFrame(window_summaries)

    metrics_df.to_csv(OUT_DIR / "trajectory_window_sensitivity_metrics.csv", index=False)
    predictions_df.to_csv(OUT_DIR / "trajectory_window_sensitivity_predictions.csv", index=False)
    window_summary_df.to_csv(OUT_DIR / "trajectory_window_sensitivity_window_counts.csv", index=False)
    plot_metrics(metrics_df)

    pivot = metrics_df.pivot(index="m", columns="window_setting", values="rmse")
    pivot = pivot[["first5", "last5", "sliding5", "random5"]]
    pivot.to_csv(OUT_DIR / "trajectory_window_sensitivity_rmse_pivot.csv")

    with (OUT_DIR / "trajectory_window_sensitivity_report.md").open("w", encoding="utf-8") as f:
        f.write("# Trajectory Window Selection Sensitivity\n\n")
        f.write("## Window counts\n\n")
        f.write(_markdown_table(window_summary_df))
        f.write("\n\n## RMSE by m and window setting\n\n")
        f.write(_markdown_table(pivot.round(4).reset_index()))
        f.write("\n\n## Full metrics\n\n")
        f.write(_markdown_table(metrics_df.round(4)))
        f.write("\n")

    print(f"Wrote trajectory window sensitivity outputs to {OUT_DIR}")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
