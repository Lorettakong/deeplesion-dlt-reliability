from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlstt_adaptive_uq_experiments.config import ExperimentConfig, TrainConfig
from nlstt_adaptive_uq_experiments.data import (
    load_deeplesion_len5_long,
    make_deeplesion_prediction_task,
    split_trajectory_ids_by_patient,
)
from nlstt_adaptive_uq_experiments.metrics import (
    conformalize_existing_intervals,
    residual_calibrate_intervals,
    summarize_predictions,
)
from nlstt_adaptive_uq_experiments.train import train_ensemble, train_single_model
from nlstt_adaptive_uq_experiments.uq_methods import (
    predict_deterministic,
    predict_ensemble,
    predict_laplace_approx,
    predict_mc_dropout,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs_nlstt_adaptive_uq_paper" / "experiment2_uq_refined"
METHOD_ORDER = ["deterministic", "mc_dropout", "deep_ensemble", "bayesian_laplace"]
METHOD_LABELS = {
    "deterministic": "Deterministic",
    "mc_dropout": "MC Dropout",
    "deep_ensemble": "Deep Ensemble",
    "bayesian_laplace": "Residual Gaussian",
}


def _format_mean_ci(mean: float, half: float) -> str:
    if np.isnan(mean):
        return "N/A"
    if np.isnan(half):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {half:.4f}"


def _summarize_repeats(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "mae",
        "rmse",
        "mape",
        "picp",
        "mpiw",
        "ece",
        "nll",
        "physics_residual_abs",
    ]
    rows = []
    group_cols = ["m", "split", "method", "variant"] if "split" in metrics.columns else ["m", "method", "variant"]
    for keys, group in metrics.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_repeats"] = int(group["repeat"].nunique())
        for col in numeric_cols:
            if col not in group.columns:
                continue
            vals = group[col].dropna().to_numpy(float)
            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_sd"] = np.nan
                row[f"{col}_ci95_half"] = np.nan
                continue
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            row[f"{col}_ci95_half"] = float(1.96 * row[f"{col}_sd"] / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _save_main_tables(summary: pd.DataFrame) -> None:
    if "split" in summary.columns:
        summary_for_tables = summary[summary["split"] == "test"].copy()
    else:
        summary_for_tables = summary.copy()
    calibrated = summary_for_tables[summary_for_tables["variant"] == "calibrated"].copy()
    table = []
    for m in [1, 2, 3, 4]:
        row = {"m": m}
        for method in METHOD_ORDER:
            sub = calibrated[(calibrated["m"] == m) & (calibrated["method"] == method)]
            if sub.empty:
                row[METHOD_LABELS[method]] = "N/A"
            else:
                item = sub.iloc[0]
                row[METHOD_LABELS[method]] = _format_mean_ci(item["rmse_mean"], item["rmse_ci95_half"])
        table.append(row)
    pd.DataFrame(table).to_csv(OUT_DIR / "table2a_calibrated_rmse.csv", index=False)

    uq_rows = []
    for variant in ["raw", "calibrated"]:
        for method in METHOD_ORDER:
            sub = summary_for_tables[(summary_for_tables["m"] == 4) & (summary_for_tables["method"] == method) & (summary_for_tables["variant"] == variant)]
            if sub.empty:
                continue
            item = sub.iloc[0]
            uq_rows.append(
                {
                    "variant": variant,
                    "method": METHOD_LABELS[method],
                    "PICP": _format_mean_ci(item.get("picp_mean", np.nan), item.get("picp_ci95_half", np.nan)),
                    "MPIW": _format_mean_ci(item.get("mpiw_mean", np.nan), item.get("mpiw_ci95_half", np.nan)),
                    "ECE": _format_mean_ci(item.get("ece_mean", np.nan), item.get("ece_ci95_half", np.nan)),
                    "NLL": _format_mean_ci(item.get("nll_mean", np.nan), item.get("nll_ci95_half", np.nan)),
                }
            )
    pd.DataFrame(uq_rows).to_csv(OUT_DIR / "table2b_m4_raw_vs_calibrated_uq.csv", index=False)


def _save_validation_selection(summary: pd.DataFrame) -> None:
    """Select the primary method for each m using validation-set RMSE only."""
    if "split" not in summary.columns:
        raise ValueError("Validation selection requires split-specific metrics.")
    candidates = summary[(summary["split"] == "validation") & (summary["variant"] == "raw")].copy()
    selection_rows = []
    audit_rows = []
    for m in [1, 2, 3, 4]:
        sub = candidates[candidates["m"] == m].sort_values("rmse_mean")
        if sub.empty:
            raise ValueError(f"No validation metrics found for m={m}.")
        for _, item in sub.iterrows():
            audit_rows.append(
                {
                    "m": m,
                    "method": METHOD_LABELS[item["method"]],
                    "validation_rmse": item["rmse_mean"],
                    "validation_rmse_ci95_half": item["rmse_ci95_half"],
                    "rank_by_validation_rmse": len([r for r in audit_rows if r.get("m") == m]) + 1,
                }
            )
        best = sub.iloc[0]
        selection_rows.append(
            {
                "m": m,
                "selected_method_id": best["method"],
                "selected_method": METHOD_LABELS[best["method"]],
                "selection_metric": "validation_rmse",
                "validation_rmse": best["rmse_mean"],
                "validation_rmse_ci95_half": best["rmse_ci95_half"],
            }
        )
    pd.DataFrame(selection_rows).to_csv(OUT_DIR / "experiment2_validation_selected_methods.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT_DIR / "experiment2_validation_selection_audit.csv", index=False)


def _plot_main(summary: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    if "split" in summary.columns:
        summary_for_plots = summary[summary["split"] == "test"].copy()
    else:
        summary_for_plots = summary.copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)

    ax = axes[0, 0]
    calibrated = summary_for_plots[summary_for_plots["variant"] == "calibrated"]
    for method in METHOD_ORDER:
        sub = calibrated[calibrated["method"] == method].sort_values("m")
        if sub.empty:
            continue
        ax.errorbar(
            sub["m"],
            sub["rmse_mean"],
            yerr=sub["rmse_ci95_half"],
            marker="o",
            linewidth=2,
            capsize=3,
            label=METHOD_LABELS[method],
        )
    ax.set_title("A. Test RMSE")
    ax.set_xlabel("Observed CT visits (m)")
    ax.set_ylabel("RMSE")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    m4 = summary_for_plots[summary_for_plots["m"] == 4].copy()
    x = np.arange(len(METHOD_ORDER))
    width = 0.36
    raw_picp = []
    cal_picp = []
    for method in METHOD_ORDER:
        raw = m4[(m4["method"] == method) & (m4["variant"] == "raw")]
        cal = m4[(m4["method"] == method) & (m4["variant"] == "calibrated")]
        raw_picp.append(raw["picp_mean"].iloc[0] if not raw.empty else np.nan)
        cal_picp.append(cal["picp_mean"].iloc[0] if not cal.empty else np.nan)
    ax.bar(x - width / 2, raw_picp, width, label="Raw")
    ax.bar(x + width / 2, cal_picp, width, label="Calibrated")
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("B. m=4 PICP")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("PICP")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    raw_mpiw = []
    cal_mpiw = []
    for method in METHOD_ORDER:
        raw = m4[(m4["method"] == method) & (m4["variant"] == "raw")]
        cal = m4[(m4["method"] == method) & (m4["variant"] == "calibrated")]
        raw_mpiw.append(raw["mpiw_mean"].iloc[0] if not raw.empty else np.nan)
        cal_mpiw.append(cal["mpiw_mean"].iloc[0] if not cal.empty else np.nan)
    ax.bar(x - width / 2, raw_mpiw, width, label="Raw")
    ax.bar(x + width / 2, cal_mpiw, width, label="Calibrated")
    ax.set_title("C. m=4 MPIW")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Mean prediction interval width")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    if not sensitivity.empty:
        for p, group in sensitivity.groupby("dropout"):
            sub = group.groupby("samples", as_index=False).agg(picp=("picp", "mean"), mpiw=("mpiw", "mean"))
            ax.plot(sub["samples"], sub["picp"], marker="o", linewidth=2, label=f"p={p:g}")
        ax.axhline(0.95, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("D. MC Dropout sensitivity, m=4")
    ax.set_xlabel("Stochastic forward passes")
    ax.set_ylabel("Raw PICP")
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.savefig(OUT_DIR / "fig_experiment2_uq_refined.png", dpi=240)
    fig.savefig(OUT_DIR / "fig_experiment2_uq_refined.pdf")
    plt.close(fig)


def _run_main_experiment(cfg: ExperimentConfig, train_cfg: TrainConfig, repeats: int) -> pd.DataFrame:
    df = load_deeplesion_len5_long(relative_log=True)
    ids = sorted(df["trajectory_id"].astype(str).unique().tolist())
    split = split_trajectory_ids_by_patient(
        df,
        ids,
        seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )
    split_counts = pd.DataFrame(
        [
            {"split": "train", "trajectories": len(split.train)},
            {"split": "validation", "trajectories": len(split.val)},
            {"split": "test", "trajectories": len(split.test)},
        ]
    )
    split_counts.to_csv(OUT_DIR / "experiment2_split_counts.csv", index=False)

    rows = []
    for repeat in range(repeats):
        repeat_seed = 91000 * repeat
        for m in [1, 2, 3, 4]:
            task_dir = OUT_DIR / f"repeat{repeat + 1}" / f"m{m}"
            task_dir.mkdir(parents=True, exist_ok=True)
            train_df = make_deeplesion_prediction_task(df, split.train, m=m)
            val_df = make_deeplesion_prediction_task(df, split.val, m=m)
            test_df = make_deeplesion_prediction_task(df, split.test, m=m)

            deterministic = train_single_model(train_df, train_cfg, method="no_physics", seed=repeat_seed + 10 + m)
            dropout = train_single_model(train_df, train_cfg, method="mc_dropout", seed=repeat_seed + 20 + m)
            ensemble = train_ensemble(train_df, train_cfg, seed=repeat_seed + 30 + m, method="heteroscedastic_ensemble")
            laplace_map = train_single_model(train_df, train_cfg, method="no_physics", seed=repeat_seed + 40 + m)

            pred_pairs = {
                "deterministic": (
                    predict_deterministic(deterministic, val_df),
                    predict_deterministic(deterministic, test_df),
                    "residual",
                ),
                "mc_dropout": (
                    predict_mc_dropout(dropout, val_df, samples=train_cfg.mc_samples),
                    predict_mc_dropout(dropout, test_df, samples=train_cfg.mc_samples),
                    "scale",
                ),
                "deep_ensemble": (
                    predict_ensemble(ensemble, val_df),
                    predict_ensemble(ensemble, test_df),
                    "scale",
                ),
                "bayesian_laplace": (
                    predict_laplace_approx(laplace_map, train_df, val_df),
                    predict_laplace_approx(laplace_map, train_df, test_df),
                    "scale",
                ),
            }

            for method, (val_pred, test_pred, cal_type) in pred_pairs.items():
                val_pred.to_csv(task_dir / f"val_pred_{method}_raw.csv", index=False)
                test_pred.to_csv(task_dir / f"pred_{method}_raw.csv", index=False)
                rows.append({"repeat": repeat + 1, "m": m, "split": "validation", "method": method, "variant": "raw", **summarize_predictions(val_pred)})
                rows.append({"repeat": repeat + 1, "m": m, "split": "test", "method": method, "variant": "raw", **summarize_predictions(test_pred)})
                if cal_type == "residual":
                    calibrated, q = residual_calibrate_intervals(val_pred, test_pred, alpha=0.05)
                else:
                    calibrated, q = conformalize_existing_intervals(val_pred, test_pred, alpha=0.05)
                calibrated["calibration_q"] = q
                calibrated.to_csv(task_dir / f"pred_{method}_calibrated.csv", index=False)
                rows.append({"repeat": repeat + 1, "m": m, "split": "test", "method": method, "variant": "calibrated", **summarize_predictions(calibrated)})
    return pd.DataFrame(rows)


def _run_dropout_sensitivity(cfg: ExperimentConfig, base_cfg: TrainConfig) -> pd.DataFrame:
    df = load_deeplesion_len5_long(relative_log=True)
    ids = sorted(df["trajectory_id"].astype(str).unique().tolist())
    split = split_trajectory_ids_by_patient(
        df,
        ids,
        seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )
    train_df = make_deeplesion_prediction_task(df, split.train, m=4)
    test_df = make_deeplesion_prediction_task(df, split.test, m=4)
    rows = []
    for repeat in range(2):
        for dropout in [0.05, 0.10, 0.20, 0.30]:
            cfg_p = base_cfg.__class__(**{**base_cfg.__dict__, "dropout": dropout})
            trained = train_single_model(train_df, cfg_p, method="mc_dropout", seed=97000 + repeat * 1000 + int(dropout * 1000))
            for samples in [30, 50, 100]:
                pred = predict_mc_dropout(trained, test_df, samples=samples)
                rows.append(
                    {
                        "repeat": repeat + 1,
                        "dropout": dropout,
                        "samples": samples,
                        **summarize_predictions(pred),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "mc_dropout_sensitivity_metrics.csv", index=False)
    summary = (
        out.groupby(["dropout", "samples"], as_index=False)
        .agg(rmse=("rmse", "mean"), picp=("picp", "mean"), mpiw=("mpiw", "mean"), ece=("ece", "mean"), nll=("nll", "mean"))
    )
    summary.to_csv(OUT_DIR / "mc_dropout_sensitivity_summary.csv", index=False)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig()
    train_cfg = TrainConfig(epochs=300, batch_size=128, hidden_dim=64, dropout=0.10, ensemble_size=5, mc_samples=50)

    metrics = _run_main_experiment(cfg, train_cfg, repeats=3)
    metrics.to_csv(OUT_DIR / "experiment2_uq_metrics.csv", index=False)
    summary = _summarize_repeats(metrics)
    summary.to_csv(OUT_DIR / "experiment2_uq_summary.csv", index=False)
    _save_main_tables(summary)
    _save_validation_selection(summary)

    sensitivity = _run_dropout_sensitivity(cfg, train_cfg)
    _plot_main(summary, sensitivity)

    print(f"Wrote refined Experiment 2 outputs to {OUT_DIR}")
    print(pd.read_csv(OUT_DIR / "table2a_calibrated_rmse.csv").to_string(index=False))
    print()
    print(pd.read_csv(OUT_DIR / "table2b_m4_raw_vs_calibrated_uq.csv").to_string(index=False))


if __name__ == "__main__":
    main()
