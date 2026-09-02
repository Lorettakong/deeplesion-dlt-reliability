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
    patient_group_kfold_ids,
    split_trajectory_ids_by_patient,
)
from nlstt_adaptive_uq_experiments.metrics import (
    patient_level_conformalize_existing_intervals,
    patient_level_residual_calibrate_intervals,
    summarize_predictions,
)
from nlstt_adaptive_uq_experiments.train import train_ensemble, train_single_model
from nlstt_adaptive_uq_experiments.uq_methods import (
    fit_gaussian_process,
    predict_deterministic,
    predict_ensemble,
    predict_residual_scale,
    predict_mc_dropout,
    predict_gaussian_process,
)


OUT_DIR = Path("outputs_nlstt_adaptive_uq_paper/experiment2_uq_refined")
METHOD_ORDER = ["deterministic", "gaussian_process", "mc_dropout", "deep_ensemble", "bayesian_laplace"]
METHOD_LABELS = {
    "deterministic": "Deterministic",
    "gaussian_process": "Gaussian Process",
    "mc_dropout": "MC Dropout",
    "deep_ensemble": "Deep Ensemble",
    "bayesian_laplace": "Gaussian residual-scale",
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
        "interval_score",
        "wis_1level",
        "physics_residual_abs",
    ]
    rows = []
    group_cols = ["m", "split", "method", "variant"] if "split" in metrics.columns else ["m", "method", "variant"]
    for keys, group in metrics.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_repeats"] = int(group["repeat"].nunique())
        if {"covered_count", "evaluation_count"}.issubset(group.columns):
            row["covered_model_repeat_evaluations"] = int(group["covered_count"].sum())
            row["total_model_repeat_evaluations"] = int(group["evaluation_count"].sum())
            if row["total_model_repeat_evaluations"] > 0:
                row["pooled_model_repeat_picp"] = (
                    row["covered_model_repeat_evaluations"] / row["total_model_repeat_evaluations"]
                )
                row["coverage_aggregation_unit"] = "trajectory_by_model_repeat"
            else:
                row["pooled_model_repeat_picp"] = np.nan
                row["coverage_aggregation_unit"] = "not_applicable_no_interval"
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
                    "Interval score": _format_mean_ci(item.get("interval_score_mean", np.nan), item.get("interval_score_ci95_half", np.nan)),
                    "WIS (one level)": _format_mean_ci(item.get("wis_1level_mean", np.nan), item.get("wis_1level_ci95_half", np.nan)),
                    "Covered / evaluated model-repeat predictions": (
                        f"{int(item.get('covered_model_repeat_evaluations', 0))}/"
                        f"{int(item.get('total_model_repeat_evaluations', 0))}"
                    ),
                }
            )
    pd.DataFrame(uq_rows).to_csv(OUT_DIR / "table2b_m4_raw_vs_calibrated_uq.csv", index=False)


def _save_development_cv_selection(cv_metrics: pd.DataFrame) -> None:
    """Select methods only inside the development patients by grouped CV."""
    candidates = (
        cv_metrics.groupby(["m", "method"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"))
    )
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
                    "development_cv_rmse": item["rmse_mean"],
                    "development_cv_rmse_sd": item["rmse_sd"],
                    "rank_by_development_cv_rmse": len([r for r in audit_rows if r.get("m") == m]) + 1,
                }
            )
        best = sub.iloc[0]
        selection_rows.append(
            {
                "m": m,
                "selected_method_id": best["method"],
                "selected_method": METHOD_LABELS[best["method"]],
                "selection_metric": "patient_grouped_development_cv_rmse",
                "development_cv_rmse": best["rmse_mean"],
                "development_cv_rmse_sd": best["rmse_sd"],
            }
        )
    pd.DataFrame(selection_rows).to_csv(OUT_DIR / "experiment2_development_cv_selected_methods.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT_DIR / "experiment2_development_cv_selection_audit.csv", index=False)


def _run_development_cv_selection(
    df: pd.DataFrame,
    development_ids: list[str],
    train_cfg: TrainConfig,
    seed: int,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Patient-grouped CV confined to training/development patients."""
    rows = []
    folds = patient_group_kfold_ids(df, development_ids, n_splits=n_splits, seed=seed)
    for fold, (fold_train_ids, fold_val_ids) in enumerate(folds, start=1):
        for m in [1, 2, 3, 4]:
            train_df = make_deeplesion_prediction_task(df, fold_train_ids, m=m)
            val_df = make_deeplesion_prediction_task(df, fold_val_ids, m=m)
            fold_seed = seed + fold * 10000 + m * 100
            deterministic = train_single_model(train_df, train_cfg, method="no_physics", seed=fold_seed + 1)
            dropout = train_single_model(train_df, train_cfg, method="mc_dropout", seed=fold_seed + 2)
            ensemble = train_ensemble(train_df, train_cfg, seed=fold_seed + 3, method="heteroscedastic_ensemble")
            laplace_map = train_single_model(train_df, train_cfg, method="no_physics", seed=fold_seed + 4)
            gp = fit_gaussian_process(train_df, seed=fold_seed + 5)
            predictions = {
                "deterministic": predict_deterministic(deterministic, val_df),
                "gaussian_process": predict_gaussian_process(gp, val_df),
                "mc_dropout": predict_mc_dropout(dropout, val_df, samples=train_cfg.mc_samples),
                "deep_ensemble": predict_ensemble(ensemble, val_df),
                "bayesian_laplace": predict_residual_scale(laplace_map, train_df, val_df),
            }
            for method, pred in predictions.items():
                rows.append({"fold": fold, "m": m, "method": method, **summarize_predictions(pred)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "experiment2_development_cv_metrics.csv", index=False)
    return out


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
    ax.set_title("A. Calibrated RMSE")
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


def _run_main_experiment(cfg: ExperimentConfig, train_cfg: TrainConfig, repeats: int, cohort_mode: str = "current") -> pd.DataFrame:
    df = load_deeplesion_len5_long(relative_log=True, cohort_mode=cohort_mode)
    ids = sorted(df["trajectory_id"].astype(str).unique().tolist())
    split = split_trajectory_ids_by_patient(
        df,
        ids,
        seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )
    patient_lookup = df.groupby("trajectory_id")["patient_id"].first().astype(str)
    split_counts = pd.DataFrame(
        [
            {"split": name, "purpose": purpose, "trajectories": len(split_ids),
             "patients": int(patient_lookup.reindex(split_ids).nunique()),
             "cohort_mode": cohort_mode}
            for name, purpose, split_ids in [
                ("development", "training_and_grouped_cv_model_selection", split.train),
                ("calibration", "patient_level_conformal_calibration_only", split.val),
                ("test", "final_evaluation_only", split.test),
            ]
        ]
    )
    split_counts.to_csv(OUT_DIR / "experiment2_split_counts.csv", index=False)

    rows = []
    gp_kernel_rows = []
    for repeat in range(repeats):
        repeat_seed = 91000 * repeat
        for m in [1, 2, 3, 4]:
            task_dir = OUT_DIR / f"repeat{repeat + 1}" / f"m{m}"
            task_dir.mkdir(parents=True, exist_ok=True)
            train_df = make_deeplesion_prediction_task(df, split.train, m=m)
            val_df = make_deeplesion_prediction_task(df, split.val, m=m)
            test_df = make_deeplesion_prediction_task(df, split.test, m=m)
            task_rows = pd.concat(
                [
                    train_df.assign(split="train"),
                    val_df.assign(split="calibration"),
                    test_df.assign(split="test"),
                ],
                ignore_index=True,
            )
            task_rows.to_csv(task_dir / "task_rows.csv", index=False)

            deterministic = train_single_model(train_df, train_cfg, method="no_physics", seed=repeat_seed + 10 + m)
            dropout = train_single_model(train_df, train_cfg, method="mc_dropout", seed=repeat_seed + 20 + m)
            ensemble = train_ensemble(train_df, train_cfg, seed=repeat_seed + 30 + m, method="heteroscedastic_ensemble")
            laplace_map = train_single_model(train_df, train_cfg, method="no_physics", seed=repeat_seed + 40 + m)
            gp = fit_gaussian_process(train_df, seed=repeat_seed + 50 + m)
            gp_estimator = gp.named_steps["gp"]
            gp_kernel_rows.append(
                {
                    "repeat": repeat + 1,
                    "m": m,
                    "seed": repeat_seed + 50 + m,
                    "optimized_kernel": str(gp_estimator.kernel_),
                    "log_marginal_likelihood": float(gp_estimator.log_marginal_likelihood_value_),
                }
            )

            pred_pairs = {
                "deterministic": (
                    predict_deterministic(deterministic, val_df),
                    predict_deterministic(deterministic, test_df),
                    "residual",
                ),
                "gaussian_process": (
                    predict_gaussian_process(gp, val_df),
                    predict_gaussian_process(gp, test_df),
                    "scale",
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
                    predict_residual_scale(laplace_map, train_df, val_df),
                    predict_residual_scale(laplace_map, train_df, test_df),
                    "scale",
                ),
            }

            for method, (val_pred, test_pred, cal_type) in pred_pairs.items():
                val_pred.to_csv(task_dir / f"val_pred_{method}_raw.csv", index=False)
                test_pred.to_csv(task_dir / f"pred_{method}_raw.csv", index=False)
                rows.append({"repeat": repeat + 1, "m": m, "split": "calibration", "method": method, "variant": "raw", **summarize_predictions(val_pred)})
                rows.append({"repeat": repeat + 1, "m": m, "split": "test", "method": method, "variant": "raw", **summarize_predictions(test_pred)})
                if cal_type == "residual":
                    calibrated, q, cal_meta = patient_level_residual_calibrate_intervals(val_pred, test_pred, alpha=0.05)
                else:
                    calibrated, q, cal_meta = patient_level_conformalize_existing_intervals(val_pred, test_pred, alpha=0.05)
                calibrated["calibration_q"] = q
                for key, value in cal_meta.items():
                    calibrated[f"calibration_{key}"] = value
                calibrated.to_csv(task_dir / f"pred_{method}_calibrated.csv", index=False)
                rows.append({"repeat": repeat + 1, "m": m, "split": "test", "method": method, "variant": "calibrated", **summarize_predictions(calibrated)})
    pd.DataFrame(gp_kernel_rows).to_csv(OUT_DIR / "gaussian_process_kernel_audit.csv", index=False)
    return pd.DataFrame(rows)


def _run_dropout_sensitivity(cfg: ExperimentConfig, base_cfg: TrainConfig, cohort_mode: str = "current") -> pd.DataFrame:
    df = load_deeplesion_len5_long(relative_log=True, cohort_mode=cohort_mode)
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


def main(cohort_mode: str = "current", repeats: int = 10, cv_folds: int = 5) -> None:
    global OUT_DIR
    if cohort_mode == "strict_unambiguous":
        OUT_DIR = Path("outputs_nlstt_adaptive_uq_paper/experiment2_uq_refined_strict_unambiguous")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig()
    train_cfg = TrainConfig(epochs=300, batch_size=128, hidden_dim=64, dropout=0.10, ensemble_size=5, mc_samples=50)

    df = load_deeplesion_len5_long(relative_log=True, cohort_mode=cohort_mode)
    ids = sorted(df["trajectory_id"].astype(str).unique().tolist())
    split = split_trajectory_ids_by_patient(
        df, ids, seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction, val_fraction=cfg.cohort.val_fraction,
    )
    cv_metrics = _run_development_cv_selection(
        df, split.train, train_cfg, seed=cfg.cohort.seed + 1205, n_splits=cv_folds
    )
    _save_development_cv_selection(cv_metrics)

    metrics = _run_main_experiment(cfg, train_cfg, repeats=repeats, cohort_mode=cohort_mode)
    metrics.to_csv(OUT_DIR / "experiment2_uq_metrics.csv", index=False)
    summary = _summarize_repeats(metrics)
    summary.to_csv(OUT_DIR / "experiment2_uq_summary.csv", index=False)
    _save_main_tables(summary)

    sensitivity = _run_dropout_sensitivity(cfg, train_cfg, cohort_mode=cohort_mode)
    _plot_main(summary, sensitivity)

    print(f"Wrote refined Experiment 2 outputs to {OUT_DIR}")
    print(pd.read_csv(OUT_DIR / "table2a_calibrated_rmse.csv").to_string(index=False))
    print()
    print(pd.read_csv(OUT_DIR / "table2b_m4_raw_vs_calibrated_uq.csv").to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-mode", choices=["current", "strict_unambiguous"], default="current")
    parser.add_argument("--repeats", type=int, default=10, help="Independent training seeds on the fixed patient split.")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    main(cohort_mode=args.cohort_mode, repeats=args.repeats, cv_folds=args.cv_folds)
