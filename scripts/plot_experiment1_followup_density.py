from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs_nlstt_adaptive_uq_paper" / "experiment1_followup_density"
EXP2_DIR = ROOT / "outputs_nlstt_adaptive_uq_paper" / "experiment2_uq_refined"
COHORT_CSV = (
    ROOT
    / "outputs_nlstt_adaptive_uq_paper"
    / "deeplesion_len5_relative_main205"
    / "cohort_deeplesion_len5_long.csv"
)

METHOD_TO_FILE = {
    "Deterministic": "pred_deterministic_calibrated.csv",
    "MC Dropout": "pred_mc_dropout_calibrated.csv",
    "Deep Ensemble": "pred_deep_ensemble_calibrated.csv",
    "Residual Gaussian": "pred_bayesian_laplace_calibrated.csv",
}

BODY_REGION_LABELS = {
    "chest_or_lung": "chest/lung",
    "abdomen_or_liver": "abdomen/liver",
    "other_or_unknown": "other",
    "lymphatic": "other",
    "pelvis": "other",
    "soft_tissue_or_breast": "other",
}


def parse_rmse_mean(value: str) -> float:
    return float(str(value).split("+/-")[0].strip())


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_pred.to_numpy() - y_true.to_numpy()) ** 2)))


def ci95_half(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def format_rmse(mean: float, ci: float) -> str:
    return f"{mean:.4f} +/- {ci:.4f}"


def selected_method_by_m() -> dict[int, str]:
    selection_path = EXP2_DIR / "experiment2_validation_selected_methods.csv"
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Missing {selection_path}. Run run_experiment2_uq_refined.py after the validation-selection patch first."
        )
    table = pd.read_csv(selection_path)
    return {int(row["m"]): str(row["selected_method"]) for _, row in table.iterrows()}


def load_test_groups() -> pd.DataFrame:
    cohort = pd.read_csv(COHORT_CSV)
    target_rows = cohort.loc[cohort["followup_index"] == 4, ["trajectory_id", "body_region_group"]].copy()
    target_rows["subgroup"] = target_rows["body_region_group"].map(BODY_REGION_LABELS).fillna("other")
    return target_rows[["trajectory_id", "subgroup"]]


def summarize(values: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in values.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mean = float(group["rmse"].mean())
        ci = ci95_half(group["rmse"])
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "N_test": int(group["N_test"].iloc[0]),
                "selected_method": group["selected_method"].iloc[0],
                "rmse": mean,
                "ci95_half": ci,
                "RMSE": format_rmse(mean, ci),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = selected_method_by_m()
    test_groups = load_test_groups()

    pooled_repeat_rows = []
    subgroup_repeat_rows = []
    for m in [1, 2, 3, 4]:
        method = selected[m]
        pred_file = METHOD_TO_FILE[method]
        for repeat in [1, 2, 3]:
            pred_path = EXP2_DIR / f"repeat{repeat}" / f"m{m}" / pred_file
            pred = pd.read_csv(pred_path)
            pred = pred.merge(test_groups, on="trajectory_id", how="left", validate="one_to_one")

            pooled_repeat_rows.append(
                {
                    "cohort": "pooled test",
                    "m": m,
                    "repeat": repeat,
                    "N_test": len(pred),
                    "selected_method": method,
                    "rmse": rmse(pred["logv_target"], pred["logv_mean"]),
                }
            )

            for subgroup, group in pred.groupby("subgroup", sort=False):
                subgroup_repeat_rows.append(
                    {
                        "subgroup": subgroup,
                        "m": m,
                        "repeat": repeat,
                        "N_test": len(group),
                        "selected_method": method,
                        "rmse": rmse(group["logv_target"], group["logv_mean"]),
                    }
                )

    pooled_by_repeat = pd.DataFrame(pooled_repeat_rows)
    subgroup_by_repeat = pd.DataFrame(subgroup_repeat_rows)

    pooled_summary = summarize(pooled_by_repeat, ["cohort", "m"])
    subgroup_summary = summarize(subgroup_by_repeat, ["subgroup", "m"])

    checks = []
    for repeat in [1, 2, 3]:
        for m in [1, 2, 3, 4]:
            pooled_rmse = pooled_by_repeat.query("repeat == @repeat and m == @m")["rmse"].iloc[0]
            subgroup_slice = subgroup_by_repeat.query("repeat == @repeat and m == @m")
            weighted = np.sqrt(
                np.sum(subgroup_slice["N_test"] * subgroup_slice["rmse"] ** 2)
                / subgroup_slice["N_test"].sum()
            )
            checks.append(
                {
                    "repeat": repeat,
                    "m": m,
                    "pooled_rmse": float(pooled_rmse),
                    "subgroup_weighted_rmse": float(weighted),
                    "abs_diff": float(abs(pooled_rmse - weighted)),
                }
            )
    consistency = pd.DataFrame(checks)

    return pooled_by_repeat, subgroup_by_repeat, pooled_summary, subgroup_summary, consistency


def plot_figure(pooled: pd.DataFrame, subgroup: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(pooled["m"], pooled["rmse"], color="#2f6da8", marker="o", linewidth=2.5)
    ax.fill_between(
        pooled["m"],
        pooled["rmse"] - pooled["ci95_half"],
        pooled["rmse"] + pooled["ci95_half"],
        color="#2f6da8",
        alpha=0.18,
    )
    for _, row in pooled.iterrows():
        ax.text(row["m"], row["rmse"] + 0.035, f"{row['rmse']:.3f}", ha="center", fontsize=9)
    ax.set_title("A. Pooled test set: validation-selected RMSE")
    ax.set_xlabel("Observed CT visits (m)")
    ax.set_ylabel("RMSE of relative log-volume")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0.22, 1.08)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    colors = {
        "chest/lung": "#3f78b5",
        "abdomen/liver": "#7ca43a",
        "other": "#8c6cc0",
    }
    for name, group in subgroup.groupby("subgroup", sort=False):
        label = f"{name} (N={int(group['N_test'].iloc[0])})"
        ax.errorbar(
            group["m"],
            group["rmse"],
            yerr=group["ci95_half"],
            marker="o",
            linewidth=2.2,
            capsize=3,
            label=label,
            color=colors[name],
        )
    ax.set_title("B. Subgroup RMSE by lesion site")
    ax.set_xlabel("Observed CT visits (m)")
    ax.set_ylabel("RMSE of relative log-volume")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0.20, 1.22)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.savefig(OUT_DIR / "fig_experiment1_followup_density.png", dpi=240)
    fig.savefig(OUT_DIR / "fig_experiment1_followup_density.pdf")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled_by_repeat, subgroup_by_repeat, pooled, subgroup, consistency = build_tables()

    pooled.to_csv(OUT_DIR / "experiment1_pooled_validation_selected_rmse.csv", index=False)
    subgroup.to_csv(OUT_DIR / "experiment1_subgroup_rmse.csv", index=False)
    pooled_by_repeat.to_csv(OUT_DIR / "experiment1_pooled_validation_selected_rmse_by_repeat.csv", index=False)
    subgroup_by_repeat.to_csv(OUT_DIR / "experiment1_subgroup_rmse_by_repeat.csv", index=False)
    consistency.to_csv(OUT_DIR / "experiment1_subgroup_consistency_check.csv", index=False)

    plot_figure(pooled, subgroup)
    print(f"Wrote validation-selected Experiment 1 figure and tables to {OUT_DIR}")
    print(f"Max pooled/subgroup consistency diff: {consistency['abs_diff'].max():.3e}")


if __name__ == "__main__":
    main()
