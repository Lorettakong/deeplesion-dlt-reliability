from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs_nlstt_adaptive_uq_paper/.mplconfig").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path("outputs_nlstt_adaptive_uq_paper/experiment0_data_quality_audit")
COHORT = Path("outputs_nlstt_adaptive_uq_paper/deeplesion_len5_relative_main205/cohort_deeplesion_len5_long.csv")


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(COHORT)
    traj = cohort[["trajectory_id", "patient_id", "body_region_group", "growth_class", "unique_studies"]].drop_duplicates()

    # Figure 0A: candidate trajectory length distribution.
    length_dist = pd.read_csv(OUT / "trajectory_length_distribution.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(length_dist["unique_scan_count"].astype(str), length_dist["components"], color="#3B6EA8")
    ax.set_xlabel("Unique CT scans per DLT component")
    ax.set_ylabel("Number of components")
    ax.set_title("A. Candidate trajectory length distribution")
    style_axes(ax)
    savefig(OUT / "fig0A_trajectory_length_distribution.png")

    # Figure 0B: main cohort body-site distribution, merging small groups into other.
    body = traj["body_region_group"].replace(
        {
            "other_or_unknown": "other",
            "lymphatic": "other",
            "pelvis": "other",
            "soft_tissue_or_breast": "other",
        }
    )
    body_counts = body.value_counts().reindex(["chest_or_lung", "abdomen_or_liver", "other"]).fillna(0)
    labels = ["chest/lung", "abdomen/liver", "other"]
    colors = ["#3B6EA8", "#7A9E3D", "#8E6BBE"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bars = ax.bar(labels, body_counts.values, color=colors)
    total = body_counts.sum()
    for bar, val in zip(bars, body_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{int(val)}\\n({val/total*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Trajectories")
    ax.set_title("B. Body-site distribution in the 205-trajectory cohort")
    style_axes(ax)
    savefig(OUT / "fig0B_body_site_distribution.png")

    # Figure 0C: split counts by patients and trajectories.
    split = pd.read_csv(OUT / "split_patient_trajectory_counts.csv")
    x = np.arange(len(split))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x - width / 2, split["patients"], width, label="Patients", color="#4C78A8")
    ax.bar(x + width / 2, split["trajectories"], width, label="Trajectories", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(["Development", "Calibration", "Test"])
    ax.set_ylabel("Count")
    ax.set_title("C. Patient-level split summary")
    ax.legend(frameon=False)
    style_axes(ax)
    savefig(OUT / "fig0C_patient_level_split.png")

    # Figure 0D: distribution comparison for V, logV, relative logV.
    variables = [
        ("V_obs_cm3_raw", "V(t), cm3"),
        ("raw_logV", "log V(t)"),
        ("logV", "relative log-volume"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for ax, (col, label) in zip(axes, variables):
        vals = cohort[col].dropna().astype(float)
        if col == "V_obs_cm3_raw":
            ax.hist(vals, bins=40, color="#3B6EA8", alpha=0.85)
            ax.set_xscale("log")
            ax.set_xlabel(label + " (log scale)")
        else:
            ax.hist(vals, bins=40, color="#3B6EA8", alpha=0.85)
            ax.set_xlabel(label)
        ax.set_ylabel("Observations")
        ax.set_title(label)
        style_axes(ax)
    fig.suptitle("D. Lesion-size target distributions", y=1.05, fontsize=13, fontweight="bold")
    savefig(OUT / "fig0D_volume_target_distributions.png")

    # Figure 0E: IQR outlier audit.
    outliers = pd.read_csv(OUT / "outlier_audit_iqr.csv")
    plot_rows = outliers[outliers["variable"].isin([
        "Raw RECIST volume proxy V(t), cm3",
        "log V(t)",
        "relative log-volume log(V(t)/V0)",
        "T4 target Raw RECIST volume proxy V(t), cm3",
        "T4 target log V(t)",
        "T4 target relative log-volume log(V(t)/V0)",
    ])].copy()
    plot_rows["label"] = [
        "All V(t)",
        "All log V(t)",
        "All relative logV",
        "T4 V(t)",
        "T4 log V(t)",
        "T4 relative logV",
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(plot_rows["label"], plot_rows["outlier_pct"], color=["#8E6BBE", "#4C78A8", "#54A24B", "#8E6BBE", "#4C78A8", "#54A24B"])
    for bar, val in zip(bars, plot_rows["outlier_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("IQR outlier rate (%)")
    ax.set_title("E. Outlier audit by target transformation")
    ax.tick_params(axis="x", rotation=25)
    style_axes(ax)
    savefig(OUT / "fig0E_outlier_audit.png")

    # Combined panel for the paper.
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    ax = axes[0, 0]
    ax.bar(length_dist["unique_scan_count"].astype(str), length_dist["components"], color="#3B6EA8")
    ax.set_xlabel("Unique CT scans per component")
    ax.set_ylabel("Components")
    ax.set_title("A. Candidate trajectory length")
    style_axes(ax)

    ax = axes[0, 1]
    bars = ax.bar(labels, body_counts.values, color=colors)
    for bar, val in zip(bars, body_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{int(val)}\\n({val/total*100:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Trajectories")
    ax.set_title("B. Body-site distribution")
    style_axes(ax)

    ax = axes[1, 0]
    x = np.arange(len(split))
    ax.bar(x - width / 2, split["patients"], width, label="Patients", color="#4C78A8")
    ax.bar(x + width / 2, split["trajectories"], width, label="Trajectories", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(["Development", "Calibration", "Test"])
    ax.set_ylabel("Count")
    ax.set_title("C. Patient-level split")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)

    ax = axes[1, 1]
    bars = ax.bar(plot_rows["label"], plot_rows["outlier_pct"], color=["#8E6BBE", "#4C78A8", "#54A24B", "#8E6BBE", "#4C78A8", "#54A24B"])
    ax.set_ylabel("IQR outlier rate (%)")
    ax.set_title("D. Outlier audit")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    style_axes(ax)
    savefig(OUT / "fig0_combined_data_quality_audit.png")

    print("Saved Experiment 0 figures to", OUT)


if __name__ == "__main__":
    main()
