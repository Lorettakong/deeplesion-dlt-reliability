from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("outputs_nlstt_adaptive_uq_paper/experiment0_data_quality_audit")
SUMMARY = Path("outputs_deeplesion_longitudinal/deeplesion_trajectory_summary.csv")
STRICT_SUMMARY = Path("outputs_deeplesion_longitudinal/deeplesion_trajectory_summary_strict.csv")
LABELS = Path("outputs_deeplesion_longitudinal/deeplesion_len5_trajectory_labels.csv")
COHORT = Path("outputs_nlstt_adaptive_uq_paper/deeplesion_len5_relative_main205/cohort_deeplesion_len5_long.csv")
SPLIT_COUNTS = Path("outputs_nlstt_adaptive_uq_paper/deeplesion_len5_relative_main205/split_patient_counts.csv")
GRAPH_AUDIT = Path("outputs_deeplesion_longitudinal/deeplesion_component_graph_audit.csv")


def q(series: pd.Series) -> dict[str, float]:
    s = series.dropna().astype(float)
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


def outlier_count(series: pd.Series) -> dict[str, float]:
    s = series.dropna().astype(float)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    mask = (s < lo) | (s > hi)
    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_fence": float(lo),
        "upper_fence": float(hi),
        "outlier_n": int(mask.sum()),
        "outlier_pct": float(mask.mean() * 100.0),
    }


def markdown_table(df: pd.DataFrame) -> str:
    view = df.copy()
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY)
    strict_summary = pd.read_csv(STRICT_SUMMARY) if STRICT_SUMMARY.exists() else pd.DataFrame()
    labels = pd.read_csv(LABELS)
    cohort = pd.read_csv(COHORT)
    split_counts = pd.read_csv(SPLIT_COUNTS)
    graph_audit = pd.read_csv(GRAPH_AUDIT) if GRAPH_AUDIT.exists() else pd.DataFrame()

    # Candidate graph-component audit.
    candidate_overview = pd.DataFrame(
        [
            {"criterion": "All DLT connected components", "trajectories": int(summary["trajectory_id"].nunique())},
            {"criterion": "Patient-consistent components", "trajectories": int((summary["patient_count"] == 1).sum())},
            {"criterion": "Trajectory length >= 3", "trajectories": int(summary["eligible_len3"].sum())},
            {"criterion": "Trajectory length = 4", "trajectories": int(((summary["unique_studies"] == 4) & (summary["patient_count"] == 1)).sum())},
            {"criterion": "Trajectory length >= 5", "trajectories": int(summary["eligible_len5"].sum())},
            {"criterion": "Trajectory length >= 8", "trajectories": int(summary["eligible_len8"].sum())},
        ]
    )
    candidate_overview.to_csv(OUT / "candidate_trajectory_overview.csv", index=False)

    if not graph_audit.empty:
        graph_summary = pd.DataFrame(
            [
                {"quantity": "Connected components", "value": len(graph_audit)},
                {"quantity": "Components with graph/study ambiguity", "value": int(graph_audit["ambiguous_component"].sum())},
                {"quantity": "Branching nodes (degree > 2)", "value": int(graph_audit["branching_node_count_degree_gt2"].sum())},
                {"quantity": "Repeated pair annotations", "value": int(graph_audit["duplicate_pair_annotations"].sum())},
                {"quantity": "Studies with multiple candidate nodes", "value": int(graph_audit["studies_with_multiple_candidate_nodes"].sum())},
                {"quantity": "Trajectories excluded by future-size/smoothness rules", "value": 0},
                {"quantity": "Strict-cohort components excluded for ambiguity", "value": int(strict_summary.get("excluded_strict_ambiguity", pd.Series(dtype=int)).sum()) if not strict_summary.empty else "rerun strict construction"},
            ]
        )
        graph_summary.to_csv(OUT / "lesion_tracking_graph_audit_summary.csv", index=False)
    else:
        graph_summary = pd.DataFrame([{"quantity": "Graph audit unavailable; rerun prepare_deeplesion_longitudinal.py", "value": "NA"}])

    # Prespecified manual audit manifest; image review fields are intentionally blank.
    manual = cohort[["trajectory_id", "patient_id", "body_region_group"]].drop_duplicates()
    manual = manual.sample(n=min(50, len(manual)), random_state=20260901).sort_values("trajectory_id")
    manual["review_same_lesion_across_visits"] = ""
    manual["review_duplicate_scan"] = ""
    manual["review_ambiguous_branch"] = ""
    manual["review_notes"] = ""
    manual.to_csv(OUT / "manual_image_audit_manifest_n50.csv", index=False)

    # Distribution of full observed trajectory length among patient-consistent components.
    patient_consistent = summary[summary["patient_count"] == 1].copy()
    bins = [0, 1, 2, 3, 4, 5, 8, 12, 20, 10**9]
    labels_bins = ["1", "2", "3", "4", "5", "6-8", "9-12", "13-20", ">20"]
    length_dist = (
        pd.cut(patient_consistent["unique_studies"], bins=bins, labels=labels_bins, right=True)
        .value_counts()
        .reindex(labels_bins)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    length_dist.columns = ["unique_scan_count", "components"]
    length_dist.to_csv(OUT / "trajectory_length_distribution.csv", index=False)

    # Main cohort audit.
    traj = cohort[["trajectory_id", "patient_id", "body_region_group", "growth_class"]].drop_duplicates()
    cohort_overview = pd.DataFrame(
        [
            {"quantity": "Trajectories", "value": int(traj["trajectory_id"].nunique())},
            {"quantity": "Patients", "value": int(traj["patient_id"].nunique())},
            {"quantity": "Time points", "value": int(len(cohort))},
            {"quantity": "Unique scans per retained trajectory", "value": "5"},
        ]
    )
    cohort_overview.to_csv(OUT / "main_cohort_overview.csv", index=False)

    per_patient = traj.groupby("patient_id", as_index=False).agg(trajectories=("trajectory_id", "nunique"))
    per_patient_dist = per_patient["trajectories"].value_counts().sort_index().reset_index()
    per_patient_dist.columns = ["trajectories_per_patient", "patients"]
    per_patient_dist.to_csv(OUT / "trajectories_per_patient_distribution.csv", index=False)

    body = traj["body_region_group"].value_counts().reset_index()
    body.columns = ["body_region_group", "trajectories"]
    body["percent"] = (body["trajectories"] / body["trajectories"].sum() * 100).round(1)
    body.to_csv(OUT / "body_site_distribution.csv", index=False)

    growth = traj["growth_class"].value_counts().reset_index()
    growth.columns = ["growth_class", "trajectories"]
    growth["percent"] = (growth["trajectories"] / growth["trajectories"].sum() * 100).round(1)
    growth.to_csv(OUT / "growth_class_distribution.csv", index=False)

    # Split by patients and trajectories.
    split_counts.to_csv(OUT / "split_patient_trajectory_counts.csv", index=False)
    # Task rows are identical across m in count, but input length differs.
    task_counts = []
    for m in [1, 2, 3, 4]:
        for _, row in split_counts.iterrows():
            task_counts.append(
                {
                    "m": m,
                    "split": row["split"],
                    "patients": int(row["patients"]),
                    "task_samples": int(row["trajectories"]),
                    "input_visits": m,
                    "target_visit": "T4",
                }
            )
    task_counts = pd.DataFrame(task_counts)
    task_counts.to_csv(OUT / "task_sample_counts_by_m.csv", index=False)

    # Volume distributions across all timepoints and baseline/final only.
    variables = [
        ("V_obs_cm3_raw", "Raw RECIST volume proxy V(t), cm3"),
        ("raw_logV", "log V(t)"),
        ("logV", "relative log-volume log(V(t)/V0)"),
    ]
    rows = []
    for col, label in variables:
        row = {"variable": label, **q(cohort[col])}
        rows.append(row)
    volume_summary = pd.DataFrame(rows).round(4)
    volume_summary.to_csv(OUT / "volume_variable_distribution.csv", index=False)

    by_visit_rows = []
    for visit, g in cohort.groupby("followup_index"):
        for col, label in variables:
            by_visit_rows.append({"visit": f"T{int(visit)}", "variable": label, **q(g[col])})
    by_visit = pd.DataFrame(by_visit_rows).round(4)
    by_visit.to_csv(OUT / "volume_variable_distribution_by_visit.csv", index=False)

    outlier_rows = []
    for col, label in variables:
        outlier_rows.append({"variable": label, **outlier_count(cohort[col])})
    # Also audit final target because all prediction tasks share T4.
    final = cohort[cohort["followup_index"] == 4]
    for col, label in variables:
        outlier_rows.append({"variable": f"T4 target {label}", **outlier_count(final[col])})
    outliers = pd.DataFrame(outlier_rows).round(4)
    outliers.to_csv(OUT / "outlier_audit_iqr.csv", index=False)

    lines = [
        "# Experiment 0: Data Quality Audit and Benchmark Characterization",
        "",
        "## Candidate Trajectory Audit",
        markdown_table(candidate_overview),
        "",
        "## Main Cohort Overview",
        markdown_table(cohort_overview),
        "",
        "## Lesion-Tracking Graph Audit",
        markdown_table(graph_summary),
        "",
        "## Patient-Level Split",
        markdown_table(split_counts),
        "",
        "## Trajectories Per Patient",
        markdown_table(per_patient_dist),
        "",
        "## Body-Site Distribution",
        markdown_table(body),
        "",
        "## Task Sample Counts by m",
        markdown_table(task_counts),
        "",
        "## Volume Variable Distribution",
        markdown_table(volume_summary),
        "",
        "## Outlier Audit (IQR Rule)",
        markdown_table(outliers),
        "",
    ]
    (OUT / "experiment0_data_quality_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "experiment0_data_quality_audit_report.md")


if __name__ == "__main__":
    main()
