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
from nlstt_adaptive_uq_experiments.metrics import summarize_predictions
from nlstt_adaptive_uq_experiments.physics import estimate_fixed_gompertz_reference
from nlstt_adaptive_uq_experiments.train import train_single_model
from nlstt_adaptive_uq_experiments.uq_methods import predict_deterministic


OUT_DIR = Path("outputs_nlstt_adaptive_uq_paper/experiment3_physics_refined")
LAMBDA_VALUES = [0.0, 0.1, 1.0, 10.0, 100.0]
LAMBDA_LABELS = {
    0.0: "No Physics",
    0.1: "lambda=0.1",
    1.0: "lambda=1",
    10.0: "lambda=10",
    100.0: "lambda=100",
}


def _seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["m", "lambda_phys", "label"], sort=False):
        row = {"m": keys[0], "lambda_phys": keys[1], "label": keys[2], "n_repeats": int(group["repeat"].nunique())}
        for col in ["rmse", "mae", "physics_residual_abs"]:
            vals = group[col].dropna().to_numpy(float)
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            row[f"{col}_ci95_half"] = float(1.96 * row[f"{col}_sd"] / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_metric(pred: pd.DataFrame, metric: str, rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float]:
    if "patient_id" not in pred.columns:
        pred = pred.copy()
        pred["patient_id"] = pred["trajectory_id"]
    patients = pred["patient_id"].astype(str).drop_duplicates().to_numpy()
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        boot = pd.concat([pred[pred["patient_id"].astype(str) == p] for p in sampled], ignore_index=True)
        y = boot["logv_target"].to_numpy(float)
        mu = boot["logv_mean"].to_numpy(float)
        if metric == "rmse":
            values.append(float(np.sqrt(np.mean((mu - y) ** 2))))
        elif metric == "mae":
            values.append(float(np.mean(np.abs(mu - y))))
        elif metric == "physics_residual_abs":
            values.append(float(np.mean(np.abs(boot["physics_residual"].to_numpy(float)))))
        else:
            raise ValueError(metric)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _patient_bootstrap_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260708)
    for keys, group in predictions.groupby(["m", "lambda_phys", "label"], sort=False):
        pooled = group.copy()
        for metric in ["rmse", "mae", "physics_residual_abs"]:
            lo, hi = _bootstrap_metric(pooled, metric, rng)
            rows.append(
                {
                    "m": keys[0],
                    "lambda_phys": keys[1],
                    "label": keys[2],
                    "metric": metric,
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                }
            )
    return pd.DataFrame(rows)


def _format(mean: float, half: float) -> str:
    return f"{mean:.4f} +/- {half:.4f}"


def _markdown_table(df: pd.DataFrame) -> str:
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


def _make_tables(summary: pd.DataFrame) -> None:
    rmse_rows = []
    resid_rows = []
    best_rows = []
    for m in [1, 2, 3, 4]:
        rmse_row = {"m": m}
        resid_row = {"m": m}
        sub = summary[summary["m"] == m].copy()
        for lam in LAMBDA_VALUES:
            item = sub[sub["lambda_phys"] == lam].iloc[0]
            rmse_row[LAMBDA_LABELS[lam]] = _format(item["rmse_mean"], item["rmse_ci95_half"])
            resid_row[LAMBDA_LABELS[lam]] = _format(item["physics_residual_abs_mean"], item["physics_residual_abs_ci95_half"])
        nonzero = sub[sub["lambda_phys"] > 0]
        best_rmse = nonzero.loc[nonzero["rmse_mean"].idxmin()]
        best_resid = nonzero.loc[nonzero["physics_residual_abs_mean"].idxmin()]
        best_rows.append(
            {
                "m": m,
                "best_lambda_by_rmse": LAMBDA_LABELS[float(best_rmse["lambda_phys"])],
                "best_rmse": best_rmse["rmse_mean"],
                "best_lambda_by_residual": LAMBDA_LABELS[float(best_resid["lambda_phys"])],
                "best_residual": best_resid["physics_residual_abs_mean"],
            }
        )
        rmse_rows.append(rmse_row)
        resid_rows.append(resid_row)
    pd.DataFrame(rmse_rows).to_csv(OUT_DIR / "table3a_fixed_lambda_rmse.csv", index=False)
    pd.DataFrame(resid_rows).to_csv(OUT_DIR / "table3b_fixed_lambda_physics_residual.csv", index=False)
    pd.DataFrame(best_rows).to_csv(OUT_DIR / "table3c_best_lambda_tradeoff.csv", index=False)


def _plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    ax = axes[0]
    for lam in LAMBDA_VALUES:
        sub = summary[summary["lambda_phys"] == lam].sort_values("m")
        ax.errorbar(
            sub["m"],
            sub["rmse_mean"],
            yerr=sub["rmse_ci95_half"],
            marker="o",
            linewidth=2,
            capsize=3,
            label=LAMBDA_LABELS[lam],
        )
    ax.set_title("A. RMSE by fixed physics weight")
    ax.set_xlabel("Observed CT visits (m)")
    ax.set_ylabel("RMSE")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    for lam in LAMBDA_VALUES:
        sub = summary[summary["lambda_phys"] == lam].sort_values("m")
        ax.errorbar(
            sub["m"],
            sub["physics_residual_abs_mean"],
            yerr=sub["physics_residual_abs_ci95_half"],
            marker="o",
            linewidth=2,
            capsize=3,
            label=LAMBDA_LABELS[lam],
        )
    ax.set_title("B. Physics residual by fixed weight")
    ax.set_xlabel("Observed CT visits (m)")
    ax.set_ylabel("Mean absolute Gompertz residual")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[2]
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    colors = {0.0: "#777777", 0.1: "#4c78a8", 1.0: "#59a14f", 10.0: "#f28e2b", 100.0: "#b07aa1"}
    for _, row in summary.iterrows():
        ax.scatter(
            row["rmse_mean"],
            row["physics_residual_abs_mean"],
            s=80,
            marker=markers[int(row["m"])],
            color=colors[float(row["lambda_phys"])],
            alpha=0.85,
        )
    for lam in LAMBDA_VALUES:
        ax.scatter([], [], color=colors[lam], label=LAMBDA_LABELS[lam])
    ax.set_title("C. Accuracy-consistency trade-off")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Mean absolute Gompertz residual")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")

    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(OUT_DIR / "fig_experiment3_physics_tradeoff.png", dpi=240)
    fig.savefig(OUT_DIR / "fig_experiment3_physics_tradeoff.pdf")
    plt.close(fig)


def _write_report(summary: pd.DataFrame) -> None:
    rmse_table = pd.read_csv(OUT_DIR / "table3a_fixed_lambda_rmse.csv")
    resid_table = pd.read_csv(OUT_DIR / "table3b_fixed_lambda_physics_residual.csv")
    best_table = pd.read_csv(OUT_DIR / "table3c_best_lambda_tradeoff.csv")
    lines = [
        "# Experiment 3: Fixed Physics-Weight Sensitivity",
        "",
        "## Mechanism",
        "",
        "The physics term is a last-observed-to-target discrete Gompertz residual on relative log-volume. "
        "For relative log-volume y(t)=log(V(t)/V(0)), the Gompertz form is dy/dt=a(c-y). "
        "Because the network predicts only the final visit, the implemented residual is "
        "r = (y_hat_T - y_last) / Delta t - a(c - 0.5*(y_last + y_hat_T)). "
        "Here a and c are estimated once from the mean trajectory of the training patients and then frozen. "
        "They are not trajectory-conditioned outputs and cannot adapt to the predicted target. Delta t is the visit-index interval.",
        "",
        "## Table 3A. RMSE",
        "",
        _markdown_table(rmse_table),
        "",
        "## Table 3B. Physics residual",
        "",
        _markdown_table(resid_table),
        "",
        "## Table 3C. Best lambda trade-off",
        "",
        _markdown_table(best_table),
        "",
        "## Interpretation",
        "",
        "Fixed physics regularization shows an accuracy-consistency trade-off. Nonzero lambda values consistently reduce the Gompertz residual relative to the no-physics model, but the lambda that minimizes residual is not necessarily the lambda that minimizes RMSE. Therefore, the physics term should be interpreted as consistency regularization rather than an unconditional accuracy booster.",
        "",
    ]
    (OUT_DIR / "experiment3_refined_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(repeats: int = 10) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig()
    train_cfg = TrainConfig(epochs=300, batch_size=128, hidden_dim=64, dropout=0.10)
    df = load_deeplesion_len5_long(relative_log=True)
    ids = sorted(df["trajectory_id"].astype(str).unique().tolist())
    split = split_trajectory_ids_by_patient(
        df,
        ids,
        seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )
    gompertz_reference = estimate_fixed_gompertz_reference(df, split.train)
    pd.DataFrame([gompertz_reference.__dict__]).to_csv(
        OUT_DIR / "gompertz_reference_training_only.csv", index=False
    )
    rows = []
    pred_rows = []
    for repeat in range(repeats):
        repeat_seed = 93000 * repeat
        for m in [1, 2, 3, 4]:
            train_df = make_deeplesion_prediction_task(df, split.train, m=m)
            test_df = make_deeplesion_prediction_task(df, split.test, m=m)
            patient_lookup = (
                test_df[["trajectory_id", "patient_id"]]
                .assign(trajectory_id=lambda x: x["trajectory_id"].astype(str))
                .set_index("trajectory_id")["patient_id"]
                .astype(str)
                .to_dict()
            )
            for lam in LAMBDA_VALUES:
                method = "no_physics" if lam == 0.0 else "fixed_pinn"
                cfg_lam = train_cfg.__class__(**{**train_cfg.__dict__, "fixed_lambda_phys": lam})
                trained = train_single_model(
                    train_df,
                    cfg_lam,
                    method=method,
                    seed=repeat_seed + m * 100 + int(lam * 10),
                    gompertz_reference=gompertz_reference,
                )
                pred = predict_deterministic(trained, test_df)
                pred["patient_id"] = pred["trajectory_id"].astype(str).map(patient_lookup)
                pred["repeat"] = repeat + 1
                pred["m"] = m
                pred["lambda_phys"] = lam
                pred["label"] = LAMBDA_LABELS[lam]
                pred_rows.append(pred)
                metric = summarize_predictions(pred)
                rows.append(
                    {
                        "repeat": repeat + 1,
                        "m": m,
                        "lambda_phys": lam,
                        "label": LAMBDA_LABELS[lam],
                        **metric,
                    }
                )
    metrics = pd.DataFrame(rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    summary = _seed_summary(metrics)
    boot = _patient_bootstrap_summary(predictions)

    metrics.to_csv(OUT_DIR / "experiment3_fixed_lambda_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / "experiment3_fixed_lambda_predictions.csv", index=False)
    summary.to_csv(OUT_DIR / "experiment3_fixed_lambda_seed_summary.csv", index=False)
    boot.to_csv(OUT_DIR / "experiment3_fixed_lambda_patient_bootstrap_ci.csv", index=False)
    _make_tables(summary)
    _plot(summary)
    _write_report(summary)
    print(f"Wrote refined Experiment 3 outputs to {OUT_DIR}")
    print(pd.read_csv(OUT_DIR / "table3a_fixed_lambda_rmse.csv").to_string(index=False))
    print()
    print(pd.read_csv(OUT_DIR / "table3b_fixed_lambda_physics_residual.csv").to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10, help="Independent training seeds on a fixed patient split.")
    args = parser.parse_args()
    main(repeats=args.repeats)
