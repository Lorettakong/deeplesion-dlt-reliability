from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta


OUTPUT_ROOT = Path("outputs_nlstt_adaptive_uq_paper")
EXP2 = OUTPUT_ROOT / "experiment2_uq_refined"
COHORT = OUTPUT_ROOT / "deeplesion_len5_relative_main205" / "cohort_deeplesion_len5_long.csv"
OUT = OUTPUT_ROOT / "experiment5_calibration_diagnostics"

METHODS = ["deterministic", "gaussian_process", "mc_dropout", "deep_ensemble", "bayesian_laplace"]
LABELS = {
    "deterministic": "Deterministic",
    "gaussian_process": "Gaussian Process",
    "mc_dropout": "MC Dropout",
    "deep_ensemble": "Deep Ensemble",
    "bayesian_laplace": "Gaussian residual-scale",
}


def interval_width(df: pd.DataFrame) -> np.ndarray:
    return df["logv_upper"].to_numpy(float) - df["logv_lower"].to_numpy(float)


def abs_error(df: pd.DataFrame) -> np.ndarray:
    return np.abs(df["logv_mean"].to_numpy(float) - df["logv_target"].to_numpy(float))


def picp(df: pd.DataFrame) -> float:
    y = df["logv_target"].to_numpy(float)
    return float(np.mean((y >= df["logv_lower"].to_numpy(float)) & (y <= df["logv_upper"].to_numpy(float))))


def coverage_flags(df: pd.DataFrame) -> np.ndarray:
    y = df["logv_target"].to_numpy(float)
    return (y >= df["logv_lower"].to_numpy(float)) & (y <= df["logv_upper"].to_numpy(float))


def clopper_pearson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    if successes <= 0:
        lower = 0.0
    else:
        lower = float(beta.ppf(alpha / 2.0, successes, n - successes + 1))
    if successes >= n:
        upper = 1.0
    else:
        upper = float(beta.ppf(1.0 - alpha / 2.0, successes + 1, n - successes))
    return lower, upper


def representative_coverage_ci(group: pd.DataFrame) -> tuple[int, int, float, float]:
    """Use one repeat for binomial CI to avoid treating repeated runs as new patients."""
    repeat = int(sorted(group["repeat"].unique())[0])
    one = group[group["repeat"] == repeat].copy()
    flags = coverage_flags(one)
    successes = int(flags.sum())
    n = int(len(flags))
    lower, upper = clopper_pearson_ci(successes, n)
    return successes, n, lower, upper


def uncertainty_error_corr(df: pd.DataFrame) -> float:
    err = abs_error(df)
    width = interval_width(df)
    if len(err) < 2 or np.std(err) < 1e-12 or np.std(width) < 1e-12:
        return float("nan")
    return float(np.corrcoef(err, width)[0, 1])


def reliability_curve(df: pd.DataFrame, levels: list[float]) -> list[float]:
    # Convert the reported 95% interval width into a Gaussian scale.
    sigma = np.maximum(interval_width(df) / (2.0 * 1.959963984540054), 1e-8)
    mu = df["logv_mean"].to_numpy(float)
    y = df["logv_target"].to_numpy(float)
    out = []
    nd = NormalDist()
    for level in levels:
        z = nd.inv_cdf(0.5 + level / 2.0)
        out.append(float(np.mean(np.abs(y - mu) <= z * sigma)))
    return out


def load_predictions(m: int = 4, variant: str = "calibrated") -> pd.DataFrame:
    rows = []
    labels = load_body_site_labels()
    repeat_dirs = sorted(
        (p for p in EXP2.glob("repeat*") if p.is_dir()),
        key=lambda p: int(p.name.removeprefix("repeat")),
    )
    for repeat_dir in repeat_dirs:
        repeat = int(repeat_dir.name.removeprefix("repeat"))
        for method in METHODS:
            path = repeat_dir / f"m{m}" / f"pred_{method}_{variant}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["repeat"] = repeat
            df["m"] = m
            df["method"] = method
            df["variant"] = variant
            rows.append(df.merge(labels, on="trajectory_id", how="left"))
    return pd.concat(rows, ignore_index=True)


def load_body_site_labels() -> pd.DataFrame:
    cohort = pd.read_csv(COHORT)
    t4 = cohort[cohort["followup_index"] == 4].copy()
    t4["subgroup"] = np.where(
        t4["body_region_group"].eq("chest_or_lung"),
        "chest/lung",
        np.where(t4["body_region_group"].eq("abdomen_or_liver"), "abdomen/liver", "other"),
    )
    return t4[["trajectory_id", "subgroup"]].drop_duplicates("trajectory_id")


def make_summary_tables(pred: pd.DataFrame) -> None:
    rows = []
    for method, group in pred.groupby("method", sort=False):
        per_repeat = []
        for repeat, g in group.groupby("repeat"):
            per_repeat.append(
                {
                    "repeat": repeat,
                    "picp": picp(g),
                    "mpiw": float(np.mean(interval_width(g))),
                    "uncertainty_error_corr": uncertainty_error_corr(g),
                    "mean_abs_error": float(np.mean(abs_error(g))),
                }
            )
        pr = pd.DataFrame(per_repeat)
        rows.append(
            {
                "Method": LABELS[method],
                "PICP": f"{pr['picp'].mean():.4f} +/- {pr['picp'].std(ddof=1):.4f}",
                "MPIW": f"{pr['mpiw'].mean():.4f} +/- {pr['mpiw'].std(ddof=1):.4f}",
                "Uncertainty-error corr": f"{pr['uncertainty_error_corr'].mean():.4f} +/- {pr['uncertainty_error_corr'].std(ddof=1):.4f}",
                "Mean absolute error": f"{pr['mean_abs_error'].mean():.4f} +/- {pr['mean_abs_error'].std(ddof=1):.4f}",
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "table5a_m4_calibration_diagnostics.csv", index=False)

    subgroup_rows = []
    for (method, subgroup), group in pred.groupby(["method", "subgroup"], sort=False):
        successes, n_independent, cp_low, cp_high = representative_coverage_ci(group)
        subgroup_rows.append(
            {
                "Method": LABELS[method],
                "Subgroup": subgroup,
                "analysis": "exploratory",
                "n_test_trajectories": n_independent,
                "covered_trajectories": successes,
                "Coverage fraction": f"{successes}/{n_independent}",
                "PICP": f"{successes / n_independent:.4f}",
                "PICP Clopper-Pearson 95% CI": f"{cp_low:.4f}-{cp_high:.4f}",
                "MPIW": f"{np.mean(interval_width(group)):.4f}",
                "MAE": f"{np.mean(abs_error(group)):.4f}",
            }
        )
    pd.DataFrame(subgroup_rows).to_csv(OUT / "table5b_subgroup_calibration.csv", index=False)

    target_rows = []
    pred = pred.copy()
    pred["target_bin"] = pd.qcut(pred["logv_target"], q=3, labels=["low y(T4)", "middle y(T4)", "high y(T4)"])
    for (method, target_bin), group in pred.groupby(["method", "target_bin"], observed=True, sort=False):
        successes, n_independent, cp_low, cp_high = representative_coverage_ci(group)
        target_rows.append(
            {
                "Method": LABELS[method],
                "Target magnitude": str(target_bin),
                "analysis": "exploratory",
                "n_test_trajectories": n_independent,
                "covered_trajectories": successes,
                "Coverage fraction": f"{successes}/{n_independent}",
                "PICP": f"{successes / n_independent:.4f}",
                "PICP Clopper-Pearson 95% CI": f"{cp_low:.4f}-{cp_high:.4f}",
                "MPIW": f"{np.mean(interval_width(group)):.4f}",
                "MAE": f"{np.mean(abs_error(group)):.4f}",
            }
        )
    pd.DataFrame(target_rows).to_csv(OUT / "table5c_target_magnitude_coverage.csv", index=False)


def make_figure(pred: pd.DataFrame) -> Path:
    representative = pred[pred["repeat"] == 1].copy()
    levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1:]),
    ]

    ax = axes[0]
    ax.plot(levels, levels, "k--", linewidth=1, label="Ideal")
    for method in METHODS:
        g = representative[representative["method"] == method]
        if g.empty:
            continue
        ax.plot(levels, reliability_curve(g, levels), marker="o", label=LABELS[method])
    ax.set_title("A. Reliability diagram, m=4")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.48, 0.97)
    ax.set_ylim(0.48, 1.02)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for method in METHODS:
        g = representative[representative["method"] == method]
        ax.scatter(interval_width(g), abs_error(g), s=24, alpha=0.65, label=LABELS[method])
    ax.set_title("B. Interval width vs absolute error")
    ax.set_xlabel("Prediction interval width")
    ax.set_ylabel("Absolute error")

    ax = axes[2]
    data = [interval_width(representative[representative["method"] == method]) for method in METHODS]
    ax.boxplot(data, tick_labels=[LABELS[m] for m in METHODS], showfliers=False)
    ax.set_title("C. Interval width distribution")
    ax.set_ylabel("Prediction interval width")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[3]
    subgroups = ["chest/lung", "abdomen/liver", "other"]
    x = np.arange(len(subgroups))
    width = 0.18
    for i, method in enumerate(METHODS):
        vals = []
        err_low = []
        err_high = []
        for subgroup in subgroups:
            g = representative[(representative["method"] == method) & (representative["subgroup"] == subgroup)]
            if len(g):
                value = picp(g)
                successes = int(coverage_flags(g).sum())
                ci_low, ci_high = clopper_pearson_ci(successes, len(g))
                vals.append(value)
                err_low.append(max(value - ci_low, 0.0))
                err_high.append(max(ci_high - value, 0.0))
            else:
                vals.append(np.nan)
                err_low.append(0.0)
                err_high.append(0.0)
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            yerr=np.asarray([err_low, err_high]),
            capsize=2,
            label=LABELS[method],
        )
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set_title("D. Subgroup PICP")
    ax.set_xticks(x)
    ax.set_xticklabels(subgroups, rotation=15)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("PICP")

    ax = axes[4]
    representative = representative.copy()
    representative["target_bin"] = pd.qcut(
        representative["logv_target"], q=3, labels=["low y(T4)", "middle y(T4)", "high y(T4)"]
    )
    bins = ["low y(T4)", "middle y(T4)", "high y(T4)"]
    x = np.arange(len(bins))
    for i, method in enumerate(METHODS):
        vals = []
        err_low = []
        err_high = []
        for target_bin in bins:
            g = representative[(representative["method"] == method) & (representative["target_bin"].astype(str) == target_bin)]
            if len(g):
                value = picp(g)
                successes = int(coverage_flags(g).sum())
                ci_low, ci_high = clopper_pearson_ci(successes, len(g))
                vals.append(value)
                err_low.append(max(value - ci_low, 0.0))
                err_high.append(max(ci_high - value, 0.0))
            else:
                vals.append(np.nan)
                err_low.append(0.0)
                err_high.append(0.0)
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            yerr=np.asarray([err_low, err_high]),
            capsize=2,
            label=LABELS[method],
        )
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set_title("E. Coverage by target magnitude")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("PICP")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    path = OUT / "fig_experiment5_calibration_diagnostics.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "fig_experiment5_calibration_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(pred: pd.DataFrame, fig_path: Path) -> Path:
    table_a = pd.read_csv(OUT / "table5a_m4_calibration_diagnostics.csv")
    table_b = pd.read_csv(OUT / "table5b_subgroup_calibration.csv")
    table_c = pd.read_csv(OUT / "table5c_target_magnitude_coverage.csv")

    def md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    lines = [
        "# Experiment 5: Calibration Diagnostics",
        "",
        "This diagnostic experiment uses the m=4 calibrated predictions from Experiment 2.",
        "",
        "## Table 5A. m=4 diagnostic summary",
        "",
        md(table_a),
        "",
        "## Table 5B. Subgroup calibration",
        "",
        md(table_b),
        "",
        "## Table 5C. Coverage by target magnitude",
        "",
        md(table_c),
        "",
        f"Figure: `{fig_path}`",
        "",
    ]
    report = OUT / "experiment5_calibration_diagnostics_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pred = load_predictions(m=4, variant="calibrated")
    make_summary_tables(pred)
    fig_path = make_figure(pred)
    print(write_report(pred, fig_path))


if __name__ == "__main__":
    main()
