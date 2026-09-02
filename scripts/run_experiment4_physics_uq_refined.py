from __future__ import annotations

from pathlib import Path
import ast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_ROOT = Path("outputs_nlstt_adaptive_uq_paper")
BASE = OUTPUT_ROOT / "deeplesion_len5_relative_physics_uq_reliability"
OUT = OUTPUT_ROOT / "experiment4_physics_uq_refined"


def fmt(mean: float, std: float | None = None, digits: int = 4) -> str:
    if std is None or pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def load_lambda_summary(lambda_value: int) -> pd.DataFrame:
    path = BASE / f"lambda_{lambda_value:g}" / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["lambda"] = lambda_value
    return df


def lambda_output_root(lambda_value: int) -> Path:
    return BASE / f"lambda_{lambda_value:g}"


def build_lambda_comparison() -> pd.DataFrame:
    """Build lambda comparisons only from the newly rerun fixed-reference outputs."""
    rows = []
    for lambda_value in [1, 10]:
        summary = load_lambda_summary(lambda_value)
        cal = summary[summary["variant"] == "calibrated"].copy()
        for m in [1, 2, 3, 4]:
            no = cal[(cal["m"] == m) & (cal["method"] == "no_physics_uq")].iloc[0]
            fixed = cal[
                (cal["m"] == m)
                & (cal["method"] == f"fixed_pinn_lambda_{lambda_value:g}_uq")
            ].iloc[0]
            rows.append(
                {
                    "lambda": f"lambda={lambda_value:g}",
                    "m": m,
                    "delta_rmse_fixed_minus_no": fixed.rmse_mean - no.rmse_mean,
                    "delta_interval_score_fixed_minus_no": fixed.interval_score_mean - no.interval_score_mean,
                    "delta_wis_fixed_minus_no": fixed.wis_1level_mean - no.wis_1level_mean,
                    "resid_ratio_fixed_over_no": fixed.physics_residual_abs_mean
                    / max(no.physics_residual_abs_mean, 1e-12),
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(BASE / "lambda1_lambda10_comparison.csv", index=False)
    return comparison


def calibrated_main_table(lambda_value: int = 1) -> pd.DataFrame:
    df = load_lambda_summary(lambda_value)
    cal = df[df["variant"] == "calibrated"].copy()
    rows = []
    for m in [1, 2, 3, 4]:
        for method_key, method_name in [
            ("no_physics_uq", "No Physics + UQ"),
            (
                f"fixed_pinn_lambda_{lambda_value:g}_uq",
                f"Gompertz-inspired regularization lambda={lambda_value:g} + UQ",
            ),
        ]:
            r = cal[(cal["m"] == m) & (cal["method"] == method_key)].iloc[0]
            rows.append(
                {
                    "m": m,
                    "Method": method_name,
                    "RMSE": fmt(r.rmse_mean, r.rmse_std),
                    "PICP": fmt(r.picp_mean, r.picp_std),
                    "MPIW": fmt(r.mpiw_mean, r.mpiw_std),
                    "Interval score": fmt(r.interval_score_mean, r.interval_score_std),
                    "WIS (one level)": fmt(r.wis_1level_mean, r.wis_1level_std),
                    "Covered/evaluated model-repeat predictions": (
                        f"{int(r.covered_model_repeat_evaluations)}/"
                        f"{int(r.total_model_repeat_evaluations)}"
                    ),
                    "Gompertz-style residual": fmt(r.physics_residual_abs_mean, r.physics_residual_abs_std),
                    "rmse_mean": r.rmse_mean,
                    "interval_score_mean": r.interval_score_mean,
                    "wis_mean": r.wis_1level_mean,
                    "resid_mean": r.physics_residual_abs_mean,
                }
            )
    return pd.DataFrame(rows)


def lambda_comparison_table() -> pd.DataFrame:
    comp = build_lambda_comparison()
    rows = []
    for lam in ["lambda=1", "lambda=10"]:
        sub = comp[comp["lambda"] == lam].copy()
        rmse_n = int((sub["delta_rmse_fixed_minus_no"] < 0).sum())
        interval_n = int((sub["delta_interval_score_fixed_minus_no"] < 0).sum())
        wis_n = int((sub["delta_wis_fixed_minus_no"] < 0).sum())
        ratio = float(sub["resid_ratio_fixed_over_no"].mean())
        rows.append(
            {
                "Fixed lambda": lam.replace("lambda=", "lambda="),
                "RMSE improvement": f"{rmse_n}/4",
                "Interval-score improvement": f"{interval_n}/4",
                "WIS improvement": f"{wis_n}/4",
                "Mean residual ratio": f"{ratio:.3f}",
                "Interpretation": (
                    "Main setting; smaller residual with more stable calibration"
                    if lam == "lambda=1"
                    else "Stronger residual reduction, but calibration/loss behavior is less uniform"
                ),
            }
        )
    return pd.DataFrame(rows)


def _parse_list(value: str) -> list[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(x) for x in ast.literal_eval(str(value))]


def _external_consistency_for_prediction(pred: pd.DataFrame, task: pd.DataFrame) -> pd.DataFrame:
    task = task[["trajectory_id", "t_obs", "logv_obs", "t_target"]].copy()
    merged = pred.merge(task, on="trajectory_id", how="left", validate="one_to_one")
    rows = []
    for _, row in merged.iterrows():
        t = np.asarray(_parse_list(row["t_obs"]), dtype=float)
        y = np.asarray(_parse_list(row["logv_obs"]), dtype=float)
        y_pred = float(row["logv_mean"])
        t_target = float(row["t_target"])
        target_dt = max(t_target - float(t[-1]), 1e-6)
        pred_slope = (y_pred - float(y[-1])) / target_dt
        endpoint_jump = abs(y_pred - float(y[-1]))
        if len(y) >= 2:
            prev_dt = max(float(t[-1] - t[-2]), 1e-6)
            prev_slope = (float(y[-1]) - float(y[-2])) / prev_dt
            slope_jump = abs(pred_slope - prev_slope)
        else:
            slope_jump = np.nan
        rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "endpoint_jump_abs": endpoint_jump,
                "slope_jump_abs": slope_jump,
            }
        )
    return pd.DataFrame(rows)


def external_consistency_table(lambda_value: int = 1) -> pd.DataFrame:
    root = lambda_output_root(lambda_value)
    rows = []
    for m in [1, 2, 3, 4]:
        repeat_rows = []
        repeat_ids = sorted(
            int(path.name.removeprefix("repeat"))
            for path in root.glob("repeat*")
            if path.is_dir() and path.name.removeprefix("repeat").isdigit()
        )
        for repeat in repeat_ids:
            task = pd.read_csv(root / f"repeat{repeat}" / f"m{m}" / "task_rows.csv")
            for method_key, method_name in [
                ("no_physics_uq", "No Physics + UQ"),
                (
                    f"fixed_pinn_lambda_{lambda_value:g}_uq",
                    f"Gompertz-inspired regularization lambda={lambda_value:g} + UQ",
                ),
            ]:
                pred_path = root / f"repeat{repeat}" / f"m{m}" / f"pred_{method_key}_calibrated.csv"
                pred = pd.read_csv(pred_path)
                diag = _external_consistency_for_prediction(pred, task)
                repeat_rows.append(
                    {
                        "repeat": repeat,
                        "m": m,
                        "Method": method_name,
                        "endpoint_jump_abs": float(diag["endpoint_jump_abs"].mean()),
                        "slope_jump_abs": float(diag["slope_jump_abs"].mean()),
                    }
                )
        rep = pd.DataFrame(repeat_rows)
        for method_name, group in rep.groupby("Method", sort=False):
            rows.append(
                {
                    "lambda": lambda_value,
                    "m": m,
                    "Method": method_name,
                    "Mean absolute endpoint jump": fmt(group["endpoint_jump_abs"].mean(), group["endpoint_jump_abs"].std(ddof=1)),
                    "Mean absolute slope jump": (
                        "N/A"
                        if group["slope_jump_abs"].isna().all()
                        else fmt(group["slope_jump_abs"].mean(), group["slope_jump_abs"].std(ddof=1))
                    ),
                    "endpoint_jump_mean": float(group["endpoint_jump_abs"].mean()),
                    "slope_jump_mean": float(group["slope_jump_abs"].mean()) if not group["slope_jump_abs"].isna().all() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def high_residual_table(lambda_value: int = 1) -> pd.DataFrame:
    path = lambda_output_root(lambda_value) / "residual_strata_metrics_summary.csv"
    strata = pd.read_csv(path)
    high = strata[
        (strata["variant"] == "calibrated") & (strata["residual_group"] == "high")
    ].copy()
    rows = []
    for m in [1, 2, 3, 4]:
        no = high[(high["m"] == m) & (high["method"] == "no_physics_uq")].iloc[0]
        fx = high[(high["m"] == m) & (high["method"] == f"fixed_pinn_lambda_{lambda_value:g}_uq")].iloc[0]
        rows.append(
            {
                "m": m,
                "High-residual RMSE No Physics": f"{no.rmse:.4f}",
                "High-residual RMSE regularized": f"{fx.rmse:.4f}",
                "High-residual interval score No Physics": f"{no.interval_score:.4f}",
                "High-residual interval score regularized": f"{fx.interval_score:.4f}",
                "High-residual WIS No Physics": f"{no.wis_1level:.4f}",
                "High-residual WIS regularized": f"{fx.wis_1level:.4f}",
                "High-residual residual No Physics": f"{no.physics_residual_abs:.4f}",
                "High-residual residual regularized": f"{fx.physics_residual_abs:.4f}",
            }
        )
    return pd.DataFrame(rows)


def make_figure() -> Path:
    main = calibrated_main_table(lambda_value=1)
    comp = build_lambda_comparison()
    ms = np.array([1, 2, 3, 4])

    no = main[main["Method"] == "No Physics + UQ"].sort_values("m")
    fx = main[main["Method"] == "Gompertz-inspired regularization lambda=1 + UQ"].sort_values("m")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(ms, no["interval_score_mean"], marker="o", label="No Physics + UQ")
    ax.plot(ms, fx["interval_score_mean"], marker="o", label="Gompertz-reg lambda=1 + UQ")
    ax.set_title("A. Conformal interval score")
    ax.set_xlabel("Observed CT visits (m)")
    ax.set_ylabel("Interval score (lower is better)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(ms, no["wis_mean"], marker="o", label="No Physics + UQ")
    ax.plot(ms, fx["wis_mean"], marker="o", label="Gompertz-reg lambda=1 + UQ")
    ax.set_title("B. Conformal WIS")
    ax.set_xlabel("Observed CT visits (m)")
    ax.set_ylabel("WIS (lower is better)")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.semilogy(ms, no["resid_mean"], marker="o", label="No Physics + UQ")
    ax.semilogy(ms, fx["resid_mean"], marker="o", label="Gompertz-reg lambda=1 + UQ")
    ax.set_title("C. Gompertz-style residual")
    ax.set_xlabel("Observed CT visits (m)")
    ax.set_ylabel("Mean absolute residual, log scale")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    deltas = []
    labels = []
    for lam in ["lambda=1", "lambda=10"]:
        sub = comp[comp["lambda"] == lam]
        labels.append(lam)
        deltas.append(
            [
                int((sub["delta_rmse_fixed_minus_no"] < 0).sum()),
                int((sub["delta_interval_score_fixed_minus_no"] < 0).sum()),
                int((sub["delta_wis_fixed_minus_no"] < 0).sum()),
            ]
        )
    x = np.arange(3)
    width = 0.34
    ax.bar(x - width / 2, deltas[0], width, label="lambda=1")
    ax.bar(x + width / 2, deltas[1], width, label="lambda=10")
    ax.set_xticks(x)
    ax.set_xticklabels(["RMSE", "Interval score", "WIS"])
    ax.set_ylim(0, 4.4)
    ax.set_ylabel("Number of improved m settings")
    ax.set_title("D. Sensitivity summary")
    ax.legend(frameon=False)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig_experiment4_physics_uq_reliability.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "fig_experiment4_physics_uq_reliability.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def write_report() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    main_l1 = calibrated_main_table(lambda_value=1)
    main_l10 = calibrated_main_table(lambda_value=10)
    sens = lambda_comparison_table()
    high = high_residual_table(lambda_value=1)
    external_l1 = external_consistency_table(lambda_value=1)
    external_l10 = external_consistency_table(lambda_value=10)

    hidden_cols = ["rmse_mean", "interval_score_mean", "wis_mean", "resid_mean"]
    main_l1.drop(columns=hidden_cols).to_csv(
        OUT / "table4a_lambda1_calibrated_uq.csv", index=False
    )
    main_l10.drop(columns=hidden_cols).to_csv(
        OUT / "table4b_lambda10_calibrated_uq.csv", index=False
    )
    sens.to_csv(OUT / "table4c_lambda_sensitivity_summary.csv", index=False)
    high.to_csv(OUT / "table4d_high_residual_stratum.csv", index=False)
    pd.concat([external_l1, external_l10], ignore_index=True).drop(
        columns=["endpoint_jump_mean", "slope_jump_mean"]
    ).to_csv(OUT / "table4e_external_trajectory_consistency.csv", index=False)
    fig_path = make_figure()

    lines = [
        "# Experiment 4: Gompertz-Inspired Regularization and UQ Reliability",
        "",
        "Main setting: Gompertz-inspired regularization lambda=1 + MC Dropout UQ versus No Physics + MC Dropout UQ.",
        "",
        "## Table 4A. Lambda=1 calibrated UQ results",
        "",
        markdown_table(main_l1.drop(columns=hidden_cols)),
        "",
        "## Table 4B. Lambda=10 calibrated UQ results",
        "",
        markdown_table(main_l10.drop(columns=hidden_cols)),
        "",
        "## Table 4C. Lambda sensitivity",
        "",
        markdown_table(sens),
        "",
        "## Table 4D. High residual stratum",
        "",
        markdown_table(high),
        "",
        "## Table 4E. External trajectory consistency diagnostics",
        "",
        markdown_table(
            pd.concat([external_l1, external_l10], ignore_index=True).drop(
                columns=["endpoint_jump_mean", "slope_jump_mean"]
            )
        ),
        "",
        f"Figure: `{fig_path}`",
        "",
    ]
    report = OUT / "experiment4_refined_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(write_report())
