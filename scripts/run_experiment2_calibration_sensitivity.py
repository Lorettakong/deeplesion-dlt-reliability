from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlstt_adaptive_uq_experiments.metrics import (
    patient_level_conformalize_existing_intervals,
    patient_level_residual_calibrate_intervals,
    summarize_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
EXP2_DIR = Path(os.environ.get(
    "DEEPLESION_EXP2_DIR",
    ROOT / "outputs_nlstt_adaptive_uq_paper" / "experiment2_uq_refined",
))
OUT_DIR = Path(os.environ.get(
    "DEEPLESION_CALIBRATION_SENSITIVITY_DIR",
    ROOT / "outputs_nlstt_adaptive_uq_paper" / "experiment2_calibration_sensitivity",
))

METHOD_ORDER = ["deterministic", "gaussian_process", "mc_dropout", "deep_ensemble", "bayesian_laplace"]
METHOD_LABELS = {
    "deterministic": "Deterministic",
    "gaussian_process": "Cohort-level Feature GP",
    "mc_dropout": "MC Dropout",
    "deep_ensemble": "Deep Ensemble",
    "bayesian_laplace": "Gaussian residual-scale",
}
CALIBRATION_TYPE = {
    "deterministic": "absolute_residual",
    "gaussian_process": "normalized_residual",
    "mc_dropout": "normalized_residual",
    "deep_ensemble": "normalized_residual",
    "bayesian_laplace": "normalized_residual",
}
ALPHAS = [0.10, 0.05]


def _wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * n)) / n) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _format_mean_ci(mean: float, half: float) -> str:
    return f"{mean:.4f} +/- {half:.4f}"


def _ci95_half(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def _finite_sample_quantile_level(n: int, alpha: float) -> tuple[float, int]:
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return rank / n, rank


def _score_distribution(val_pred: pd.DataFrame, method: str) -> np.ndarray:
    y = val_pred["logv_target"].to_numpy(float)
    mean = val_pred["logv_mean"].to_numpy(float)
    if CALIBRATION_TYPE[method] == "absolute_residual":
        return np.abs(y - mean)
    sigma = np.clip(
        (val_pred["logv_upper"].to_numpy(float) - val_pred["logv_lower"].to_numpy(float))
        / (2.0 * 1.959963984540054),
        1e-6,
        None,
    )
    return np.abs(y - mean) / sigma


def _raw_metrics_for_method(pred: pd.DataFrame, method: str) -> dict[str, float]:
    if method == "deterministic":
        reg = summarize_predictions(pred)
        return {
            "raw_rmse": reg["rmse"],
            "raw_picp": np.nan,
            "raw_mpiw": np.nan,
            "raw_ece": np.nan,
            "raw_nll": np.nan,
        }
    raw = summarize_predictions(pred)
    return {
        "raw_rmse": raw["rmse"],
        "raw_picp": raw.get("picp", np.nan),
        "raw_mpiw": raw.get("mpiw", np.nan),
        "raw_ece": raw.get("ece", np.nan),
        "raw_nll": raw.get("nll", np.nan),
    }


def _calibrate(
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    method: str,
    alpha: float,
) -> tuple[pd.DataFrame, float, dict[str, float | int | str]]:
    if CALIBRATION_TYPE[method] == "absolute_residual":
        return patient_level_residual_calibrate_intervals(val_pred, test_pred, alpha=alpha)
    return patient_level_conformalize_existing_intervals(val_pred, test_pred, alpha=alpha)


def build_audit_table() -> pd.DataFrame:
    rows = []
    repeat_ids = sorted(
        int(path.name.removeprefix("repeat"))
        for path in EXP2_DIR.glob("repeat*")
        if path.is_dir() and path.name.removeprefix("repeat").isdigit()
    )
    if not repeat_ids:
        raise FileNotFoundError(f"No repeat directories found in {EXP2_DIR}")
    for repeat in repeat_ids:
        for m in [1, 2, 3, 4]:
            task_dir = EXP2_DIR / f"repeat{repeat}" / f"m{m}"
            for method in METHOD_ORDER:
                val_path = task_dir / f"val_pred_{method}_raw.csv"
                test_path = task_dir / f"pred_{method}_raw.csv"
                if not val_path.exists() or not test_path.exists():
                    raise FileNotFoundError(f"Missing raw predictions for repeat={repeat}, m={m}, method={method}")
                val_pred = pd.read_csv(val_path)
                test_pred = pd.read_csv(test_path)
                scores = _score_distribution(val_pred, method)
                raw = _raw_metrics_for_method(test_pred, method)

                for alpha in ALPHAS:
                    calibrated, q, calibration_metadata = _calibrate(
                        val_pred, test_pred, method, alpha=alpha
                    )
                    cal = summarize_predictions(calibrated)
                    n_val = len(val_pred)
                    n_test = len(test_pred)
                    q_level = float(calibration_metadata["conformal_level"])
                    q_rank = int(calibration_metadata["conformal_rank"])
                    covered = (
                        (calibrated["logv_target"].to_numpy(float) >= calibrated["logv_lower"].to_numpy(float))
                        & (calibrated["logv_target"].to_numpy(float) <= calibrated["logv_upper"].to_numpy(float))
                    )
                    k = int(covered.sum())
                    wilson_low, wilson_high = _wilson_interval(k, n_test)
                    rows.append(
                        {
                            "repeat": repeat,
                            "m": m,
                            "method": METHOD_LABELS[method],
                            "method_id": method,
                            "calibration_type": CALIBRATION_TYPE[method],
                            "alpha": alpha,
                            "nominal_coverage": 1.0 - alpha,
                            "n_val": n_val,
                            "n_calibration_patients": int(calibration_metadata["n_calibration_patients"]),
                            "n_test": n_test,
                            "finite_sample_quantile_level": q_level,
                            "finite_sample_quantile_rank": q_rank,
                            "q_score": q,
                            "score_min": float(np.min(scores)),
                            "score_median": float(np.median(scores)),
                            "score_p90": float(np.quantile(scores, 0.90, method="higher")),
                            "score_max": float(np.max(scores)),
                            **raw,
                            "calibrated_picp": cal["picp"],
                            "calibrated_mpiw": cal["mpiw"],
                            "calibrated_ece": cal["ece"],
                            "calibrated_nll": cal["nll"],
                            "covered_count": k,
                            "picp_wilson95_low": wilson_low,
                            "picp_wilson95_high": wilson_high,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_audit(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "q_score",
        "raw_picp",
        "raw_mpiw",
        "calibrated_picp",
        "calibrated_mpiw",
        "calibrated_ece",
        "calibrated_nll",
        "picp_wilson95_low",
        "picp_wilson95_high",
    ]
    group_cols = ["m", "method", "calibration_type", "alpha", "nominal_coverage", "n_val", "n_calibration_patients", "n_test", "finite_sample_quantile_level", "finite_sample_quantile_rank"]
    for keys, group in audit.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_repeats"] = int(group["repeat"].nunique())
        for metric in metrics:
            vals = group[metric].dropna()
            if vals.empty:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_ci95_half"] = np.nan
                row[metric.upper()] = "N/A"
            else:
                mean = float(vals.mean())
                half = _ci95_half(vals)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci95_half"] = half
                row[metric.upper()] = _format_mean_ci(mean, half)
        rows.append(row)
    return pd.DataFrame(rows)


def selected_method_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    selection = pd.read_csv(EXP2_DIR / "experiment2_validation_selected_methods.csv")
    rows = []
    for _, sel in selection.iterrows():
        m = int(sel["m"])
        method = sel["selected_method"]
        sub = summary[(summary["m"] == m) & (summary["method"] == method)].copy()
        sub.insert(2, "selected_by_validation_rmse", True)
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def plot_m4_sensitivity(summary: pd.DataFrame) -> None:
    m4 = summary[summary["m"] == 4].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(METHOD_ORDER))
    width = 0.35
    for offset, alpha in [(-width / 2, 0.10), (width / 2, 0.05)]:
        sub = m4[m4["alpha"] == alpha].set_index("method").loc[[METHOD_LABELS[m] for m in METHOD_ORDER]]
        ax.bar(x + offset, sub["calibrated_picp_mean"], width=width, label=f"{int((1-alpha)*100)}% target")
        yerr_low = np.clip(sub["calibrated_picp_mean"] - sub["picp_wilson95_low_mean"], 0.0, None)
        yerr_high = np.clip(sub["picp_wilson95_high_mean"] - sub["calibrated_picp_mean"], 0.0, None)
        ax.errorbar(
            x + offset,
            sub["calibrated_picp_mean"],
            yerr=[yerr_low, yerr_high],
            fmt="none",
            color="black",
            linewidth=1,
            capsize=3,
        )
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1, alpha=0.45)
    ax.axhline(0.90, color="black", linestyle=":", linewidth=1, alpha=0.45)
    ax.set_ylim(0.75, 1.03)
    ax.set_title("A. m=4 calibrated PICP with Wilson CI")
    ax.set_ylabel("PICP")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    for method in METHOD_ORDER:
        sub = m4[m4["method"] == METHOD_LABELS[method]].sort_values("nominal_coverage")
        ax.plot(
            sub["calibrated_mpiw_mean"],
            sub["calibrated_picp_mean"],
            marker="o",
            linewidth=2,
            label=METHOD_LABELS[method],
        )
        for _, row in sub.iterrows():
            ax.text(row["calibrated_mpiw_mean"], row["calibrated_picp_mean"] - 0.012, f"{int(row['nominal_coverage']*100)}%", fontsize=8, ha="center")
    ax.set_title("B. m=4 coverage-width trade-off")
    ax.set_xlabel("Mean prediction interval width")
    ax.set_ylabel("PICP")
    ax.set_ylim(0.75, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(OUT_DIR / "fig_experiment2_calibration_sensitivity_m4.png", dpi=240)
    fig.savefig(OUT_DIR / "fig_experiment2_calibration_sensitivity_m4.pdf")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = build_audit_table()
    audit.to_csv(OUT_DIR / "experiment2_calibration_audit_by_repeat.csv", index=False)
    summary = summarize_audit(audit)
    summary.to_csv(OUT_DIR / "experiment2_calibration_sensitivity_summary.csv", index=False)
    selected = selected_method_sensitivity(summary)
    selected.to_csv(OUT_DIR / "experiment2_validation_selected_calibration_sensitivity.csv", index=False)
    plot_m4_sensitivity(summary)

    print(f"Wrote calibration audit and sensitivity outputs to {OUT_DIR}")
    print()
    print("Validation-selected methods, 95% target:")
    show_cols = [
        "m",
        "method",
        "nominal_coverage",
        "finite_sample_quantile_rank",
        "finite_sample_quantile_level",
        "Q_SCORE",
        "CALIBRATED_PICP",
        "CALIBRATED_MPIW",
        "PICP_WILSON95_LOW",
        "PICP_WILSON95_HIGH",
    ]
    print(selected[selected["alpha"] == 0.05][show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
