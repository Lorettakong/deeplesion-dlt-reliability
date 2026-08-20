from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlstt_adaptive_uq_experiments.config import TrainConfig
from nlstt_adaptive_uq_experiments.data import make_dense_prediction_task
from nlstt_adaptive_uq_experiments.metrics import summarize_predictions
from nlstt_adaptive_uq_experiments.train import train_single_model
from nlstt_adaptive_uq_experiments.uq_methods import predict_deterministic
from run_trajectory_window_sensitivity import (
    OUT_DIR,
    _load_main_trajectory_splits,
    _prepare_raw,
    build_windows,
)


MLP_OUT_DIR = OUT_DIR / "mlp_no_physics"


def _assign_split(windows: pd.DataFrame, trajectory_splits: dict[str, set[str]]) -> pd.DataFrame:
    ids = (
        windows.groupby("trajectory_id", as_index=False)
        .agg(patient_id=("patient_id", "first"), original_trajectory_id=("original_trajectory_id", "first"))
    )
    ids["split"] = "unused"
    ids["original_trajectory_id"] = ids["original_trajectory_id"].astype(str)
    for split, trajectory_ids in trajectory_splits.items():
        ids.loc[ids["original_trajectory_id"].isin(trajectory_ids), "split"] = split
    return ids


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    arr = values.to_numpy(float)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return mean, float("nan"), float("nan")
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _plot(summary: pd.DataFrame) -> None:
    order = ["first5", "last5", "sliding5", "random5"]
    labels = {
        "first5": "First-5",
        "last5": "Last-5",
        "sliding5": "Sliding-5",
        "random5": "Random-5",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for setting in order:
        sub = summary[summary["window_setting"] == setting].sort_values("m")
        if sub.empty:
            continue
        ax.plot(sub["m"], sub["rmse_mean"], marker="o", linewidth=2, label=labels[setting])
        ax.fill_between(sub["m"], sub["rmse_ci_low"], sub["rmse_ci_high"], alpha=0.12)
    ax.set_xlabel("Observed visits used for prediction (m)")
    ax.set_ylabel("RMSE of relative log-volume")
    ax.set_title("Trajectory selection sensitivity with no-physics MLP")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(MLP_OUT_DIR / "fig_trajectory_window_sensitivity_mlp_rmse.png", dpi=220)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: f"{x:.4f}")
    lines = [
        "| " + " | ".join(map(str, text.columns)) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in text.to_numpy())
    return "\n".join(lines)


def main() -> None:
    MLP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = _prepare_raw()
    trajectory_splits = _load_main_trajectory_splits()
    train_cfg = TrainConfig(epochs=300, batch_size=128, hidden_dim=64, dropout=0.0, fixed_lambda_phys=0.0)

    rows = []
    prediction_frames = []
    window_counts = []
    for setting in ["first5", "last5", "sliding5", "random5"]:
        windows = build_windows(raw, setting=setting)
        split_ids = _assign_split(windows, trajectory_splits)
        split_ids.to_csv(MLP_OUT_DIR / f"split_ids_{setting}.csv", index=False)
        window_counts.append(
            {
                "window_setting": setting,
                "n_windows": int(split_ids.shape[0]),
                "n_train": int((split_ids["split"] == "train").sum()),
                "n_validation": int((split_ids["split"] == "valid").sum()),
                "n_test": int((split_ids["split"] == "test").sum()),
                "n_patients": int(split_ids["patient_id"].nunique()),
                "n_original_trajectories": int(split_ids["original_trajectory_id"].nunique()),
            }
        )
        train_ids = split_ids.loc[split_ids["split"] == "train", "trajectory_id"].astype(str).tolist()
        test_ids = split_ids.loc[split_ids["split"] == "test", "trajectory_id"].astype(str).tolist()
        for m in [1, 2, 3, 4]:
            train_df = make_dense_prediction_task(windows, train_ids, k=m)
            test_df = make_dense_prediction_task(windows, test_ids, k=m)
            for repeat in range(5):
                seed = 20260707 + repeat * 1000 + m * 10
                trained = train_single_model(train_df, train_cfg, method="no_physics", seed=seed)
                pred = predict_deterministic(trained, test_df)
                pred["window_setting"] = setting
                pred["m"] = m
                pred["repeat"] = repeat + 1
                prediction_frames.append(pred)
                metrics = summarize_predictions(pred)
                rows.append(
                    {
                        "window_setting": setting,
                        "m": m,
                        "repeat": repeat + 1,
                        "n_train_windows": int(len(train_df)),
                        "n_test_windows": int(len(test_df)),
                        **metrics,
                    }
                )

    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    window_count_df = pd.DataFrame(window_counts)
    summary_rows = []
    for (setting, m), group in metrics.groupby(["window_setting", "m"], sort=False):
        rmse_mean, rmse_low, rmse_high = _mean_ci(group["rmse"])
        mae_mean, mae_low, mae_high = _mean_ci(group["mae"])
        summary_rows.append(
            {
                "window_setting": setting,
                "m": int(m),
                "n_train_windows": int(group["n_train_windows"].iloc[0]),
                "n_test_windows": int(group["n_test_windows"].iloc[0]),
                "rmse_mean": rmse_mean,
                "rmse_ci_low": rmse_low,
                "rmse_ci_high": rmse_high,
                "mae_mean": mae_mean,
                "mae_ci_low": mae_low,
                "mae_ci_high": mae_high,
            }
        )
    summary = pd.DataFrame(summary_rows)
    pivot = summary.pivot(index="m", columns="window_setting", values="rmse_mean")
    pivot = pivot[["first5", "last5", "sliding5", "random5"]]

    metrics.to_csv(MLP_OUT_DIR / "trajectory_window_sensitivity_mlp_metrics_repeats.csv", index=False)
    summary.to_csv(MLP_OUT_DIR / "trajectory_window_sensitivity_mlp_summary.csv", index=False)
    predictions.to_csv(MLP_OUT_DIR / "trajectory_window_sensitivity_mlp_predictions.csv", index=False)
    window_count_df.to_csv(MLP_OUT_DIR / "trajectory_window_sensitivity_mlp_window_counts.csv", index=False)
    pivot.to_csv(MLP_OUT_DIR / "trajectory_window_sensitivity_mlp_rmse_pivot.csv")
    _plot(summary)

    with (MLP_OUT_DIR / "trajectory_window_sensitivity_mlp_report.md").open("w", encoding="utf-8") as f:
        f.write("# Trajectory Window Selection Sensitivity: No-Physics MLP\n\n")
        f.write("## Window counts\n\n")
        f.write(_markdown_table(window_count_df))
        f.write("\n\n## RMSE summary\n\n")
        f.write(_markdown_table(pivot.reset_index()))
        f.write("\n\n## Full summary\n\n")
        f.write(_markdown_table(summary))
        f.write("\n")

    print(f"Wrote MLP trajectory-window sensitivity outputs to {MLP_OUT_DIR}")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
