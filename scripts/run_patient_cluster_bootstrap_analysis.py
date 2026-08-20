from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nlstt_adaptive_uq_experiments.config import ExperimentConfig
from nlstt_adaptive_uq_experiments.data import load_deeplesion_len5_long, split_trajectory_ids_by_patient
from nlstt_adaptive_uq_experiments.metrics import summarize_predictions


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_nlstt_adaptive_uq_paper" / "patient_cluster_bootstrap"
EXP2 = ROOT / "outputs_nlstt_adaptive_uq_paper" / "experiment2_uq_refined"
EXP1_TRAD = ROOT / "outputs_nlstt_adaptive_uq_paper" / "experiment1_traditional_baselines"

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

METRICS = ["rmse", "mae", "picp", "mpiw", "ece", "nll", "physics_residual_abs"]


def ci_text(mean: float, lo: float, hi: float, digits: int = 4) -> str:
    if np.isnan(mean):
        return ""
    return f"{mean:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def patient_map_and_labels() -> pd.DataFrame:
    df = load_deeplesion_len5_long(relative_log=True)
    rows = (
        df.sort_values("followup_index")
        .groupby("trajectory_id", as_index=False)
        .agg(
            patient_id=("patient_id", "first"),
            body_region_group=("body_region_group", "first"),
            body_site_primary=("body_site_primary", "first"),
            lesion_type_primary=("lesion_type_primary", "first"),
        )
    )
    rows["trajectory_id"] = rows["trajectory_id"].astype(str)
    rows["patient_id"] = rows["patient_id"].astype(str)
    rows["subgroup"] = rows["body_region_group"].map(BODY_REGION_LABELS).fillna("other")
    return rows


def selected_methods() -> dict[int, str]:
    table = pd.read_csv(EXP2 / "experiment2_validation_selected_methods.csv")
    return {int(row["m"]): str(row["selected_method"]) for _, row in table.iterrows()}


def add_patient_info(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    pred["trajectory_id"] = pred["trajectory_id"].astype(str)
    return pred.merge(
        labels[["trajectory_id", "patient_id", "subgroup", "body_site_primary", "lesion_type_primary"]],
        on="trajectory_id",
        how="left",
        validate="many_to_one",
    )


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    vals = summarize_predictions(frame)
    return {metric: float(vals[metric]) if metric in vals else np.nan for metric in METRICS}


def prepare_bootstrap_arrays(frame: pd.DataFrame) -> dict:
    patients = frame["patient_id"].dropna().astype(str).drop_duplicates().to_numpy()
    patient_indices = [
        np.flatnonzero(frame["patient_id"].astype(str).to_numpy() == str(pid))
        for pid in patients
    ]
    arrays = {
        "patients": patients,
        "patient_indices": patient_indices,
        "y": frame["logv_target"].to_numpy(float),
        "mean": frame["logv_mean"].to_numpy(float),
        "lower": frame["logv_lower"].to_numpy(float) if "logv_lower" in frame.columns else None,
        "upper": frame["logv_upper"].to_numpy(float) if "logv_upper" in frame.columns else None,
        "physics": frame["physics_residual_abs"].to_numpy(float)
        if "physics_residual_abs" in frame.columns
        else (
            np.abs(frame["physics_residual"].to_numpy(float))
            if "physics_residual" in frame.columns
            else None
        ),
    }
    return arrays


def metric_values_from_indices(arrays: dict, idx: np.ndarray | None = None) -> dict[str, float]:
    if idx is None:
        idx = np.arange(len(arrays["y"]))
    y = arrays["y"][idx]
    mean = arrays["mean"][idx]
    err = mean - y
    out = {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "picp": np.nan,
        "mpiw": np.nan,
        "ece": np.nan,
        "nll": np.nan,
        "physics_residual_abs": np.nan,
    }
    lower = arrays["lower"]
    upper = arrays["upper"]
    if lower is not None and upper is not None:
        lo = lower[idx]
        hi = upper[idx]
        covered = (y >= lo) & (y <= hi)
        out["picp"] = float(np.mean(covered))
        out["mpiw"] = float(np.mean(hi - lo))
        sigma = np.clip((hi - lo) / (2.0 * 1.959963984540054), 1e-6, None)
        nll = 0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * ((y - mean) / sigma) ** 2
        out["nll"] = float(np.mean(nll))
        z_values = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.959963984540054])
        nominal = np.array([0.1974, 0.3829, 0.5467, 0.6827, 0.7887, 0.8664, 0.9199, 0.9500])
        abs_err = np.abs(y - mean)
        empirical = np.array([float(np.mean(abs_err <= z * sigma)) for z in z_values])
        out["ece"] = float(np.mean(np.abs(empirical - nominal)))
    if arrays["physics"] is not None:
        out["physics_residual_abs"] = float(np.mean(arrays["physics"][idx]))
    return out


def cluster_bootstrap_frames(
    frames: list[pd.DataFrame],
    n_boot: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    prepared = [prepare_bootstrap_arrays(frame) for frame in frames]
    point_rows = [metric_values_from_indices(arrays) for arrays in prepared]
    point = {metric: float(np.nanmean([row[metric] for row in point_rows])) for metric in METRICS}

    rng = np.random.default_rng(seed)
    boot_rows = []
    for _ in range(n_boot):
        repeat_vals = []
        for arrays in prepared:
            sampled_patient_idx = rng.integers(0, len(arrays["patients"]), size=len(arrays["patients"]))
            sampled_rows = np.concatenate([arrays["patient_indices"][j] for j in sampled_patient_idx])
            repeat_vals.append(metric_values_from_indices(arrays, sampled_rows))
        boot_rows.append({metric: float(np.nanmean([row[metric] for row in repeat_vals])) for metric in METRICS})
    boot = pd.DataFrame(boot_rows)
    ci = {}
    for metric in METRICS:
        vals = boot[metric].dropna().to_numpy(float)
        ci[metric] = (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if len(vals) else (np.nan, np.nan)
    return point, ci


def summarize_cluster_result(
    rows: list[dict],
    label_cols: dict,
    frames: list[pd.DataFrame],
    n_boot: int,
    seed: int,
) -> None:
    point, ci = cluster_bootstrap_frames(frames, n_boot=n_boot, seed=seed)
    row = dict(label_cols)
    row["N_test_trajectories"] = int(np.mean([len(frame) for frame in frames]))
    row["N_test_patients"] = int(np.mean([frame["patient_id"].nunique() for frame in frames]))
    for metric in METRICS:
        lo, hi = ci[metric]
        row[f"{metric}_mean"] = point[metric]
        row[f"{metric}_ci95_low"] = lo
        row[f"{metric}_ci95_high"] = hi
        row[metric.upper() if metric != "picp" else "PICP"] = ci_text(point[metric], lo, hi)
    rows.append(row)


def load_selected_neural_predictions(labels: pd.DataFrame, m: int) -> list[pd.DataFrame]:
    method = selected_methods()[m]
    pred_file = METHOD_TO_FILE[method]
    frames = []
    for repeat in [1, 2, 3]:
        pred = pd.read_csv(EXP2 / f"repeat{repeat}" / f"m{m}" / pred_file)
        pred = add_patient_info(pred, labels)
        pred["repeat"] = repeat
        pred["selected_method"] = method
        frames.append(pred)
    return frames


def experiment1_neural_and_subgroups(labels: pd.DataFrame, n_boot: int) -> None:
    selected = selected_methods()
    pooled_rows = []
    subgroup_rows = []
    for m in [1, 2, 3, 4]:
        frames = load_selected_neural_predictions(labels, m)
        summarize_cluster_result(
            pooled_rows,
            {"m": m, "method": selected[m], "analysis": "pooled test"},
            frames,
            n_boot=n_boot,
            seed=21000 + m,
        )
        for subgroup in ["chest/lung", "abdomen/liver", "other"]:
            sub_frames = [frame[frame["subgroup"] == subgroup].copy() for frame in frames]
            summarize_cluster_result(
                subgroup_rows,
                {"m": m, "method": selected[m], "subgroup": subgroup, "analysis": "subgroup exploratory"},
                sub_frames,
                n_boot=n_boot,
                seed=22000 + m * 10 + len(subgroup),
            )
    pd.DataFrame(pooled_rows).to_csv(OUT / "experiment1_neural_patient_cluster_bootstrap.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(OUT / "experiment1_subgroup_patient_cluster_bootstrap.csv", index=False)


def load_traditional_prediction(labels: pd.DataFrame, m: int, method: str) -> pd.DataFrame:
    pred = pd.read_csv(EXP1_TRAD / "experiment1_traditional_baseline_predictions.csv")
    pred = pred[(pred["m"] == m) & (pred["method"] == method)].copy()
    return add_patient_info(pred, labels)


def paired_rmse_difference(
    neural_frames: list[pd.DataFrame],
    baseline: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    merged_frames = []
    for frame in neural_frames:
        merged = frame[["trajectory_id", "patient_id", "logv_target", "logv_mean"]].merge(
            baseline[["trajectory_id", "logv_mean"]].rename(columns={"logv_mean": "baseline_mean"}),
            on="trajectory_id",
            how="inner",
            validate="one_to_one",
        )
        merged_frames.append(merged)

    def patient_sse(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
        tmp = frame[["patient_id", "logv_target", pred_col]].copy()
        tmp["sq_error"] = (tmp[pred_col].to_numpy(float) - tmp["logv_target"].to_numpy(float)) ** 2
        return tmp.groupby("patient_id", as_index=False).agg(sse=("sq_error", "sum"), n=("sq_error", "size"))

    neural_aggs = [patient_sse(mf, "logv_mean") for mf in merged_frames]
    baseline_aggs = [patient_sse(mf, "baseline_mean") for mf in merged_frames]

    def rmse_from_agg(agg: pd.DataFrame) -> float:
        return float(np.sqrt(agg["sse"].sum() / agg["n"].sum()))

    point = float(np.mean([rmse_from_agg(nagg) - rmse_from_agg(bagg) for nagg, bagg in zip(neural_aggs, baseline_aggs)]))
    patients = merged_frames[0]["patient_id"].drop_duplicates().astype(str).to_numpy()
    neural_maps = [{str(row["patient_id"]): (float(row["sse"]), int(row["n"])) for _, row in agg.iterrows()} for agg in neural_aggs]
    baseline_maps = [{str(row["patient_id"]): (float(row["sse"]), int(row["n"])) for _, row in agg.iterrows()} for agg in baseline_aggs]
    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        repeat_deltas = []
        for neural_map, baseline_map in zip(neural_maps, baseline_maps):
            n_sse = 0.0
            b_sse = 0.0
            total_n = 0
            for pid in sampled:
                ns, nn = neural_map[str(pid)]
                bs, bn = baseline_map[str(pid)]
                n_sse += ns
                b_sse += bs
                total_n += nn
                if nn != bn:
                    raise ValueError("Paired bootstrap patient counts do not match.")
            repeat_deltas.append(float(np.sqrt(n_sse / total_n) - np.sqrt(b_sse / total_n)))
        deltas.append(float(np.mean(repeat_deltas)))
    return point, float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def experiment1_paired_differences(labels: pd.DataFrame, n_boot: int) -> None:
    methods_by_m = {
        1: ["Training-set mean target", "Training-set median target", "LOCF", "Ridge trajectory regression"],
        2: [
            "Training-set mean target",
            "Training-set median target",
            "LOCF",
            "Ridge trajectory regression",
            "Least-squares linear extrapolation",
            "Recent-two linear extrapolation",
            "Gaussian process regression",
        ],
        3: [
            "LOCF",
            "Ridge trajectory regression",
            "Least-squares linear extrapolation",
            "Recent-two linear extrapolation",
            "Gaussian process regression",
            "Gompertz curve fit",
            "Logistic curve fit",
        ],
        4: [
            "LOCF",
            "Ridge trajectory regression",
            "Least-squares linear extrapolation",
            "Recent-two linear extrapolation",
            "Gaussian process regression",
            "Gompertz curve fit",
            "Logistic curve fit",
        ],
    }
    rows = []
    for m, methods in methods_by_m.items():
        neural_frames = load_selected_neural_predictions(labels, m)
        for method in methods:
            baseline = load_traditional_prediction(labels, m, method)
            point, lo, hi = paired_rmse_difference(neural_frames, baseline, n_boot=n_boot, seed=31000 + m * 100 + len(method))
            rows.append(
                {
                    "m": m,
                    "comparison": f"Neural/UQ minus {method}",
                    "delta_rmse_mean": point,
                    "delta_rmse_ci95_low": lo,
                    "delta_rmse_ci95_high": hi,
                    "Delta_RMSE": ci_text(point, lo, hi),
                    "interpretation": "Negative values favor neural/UQ; positive values favor the traditional baseline.",
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "experiment1_paired_rmse_difference_cluster_bootstrap.csv", index=False)


def experiment2_m4_uq_cluster_bootstrap(labels: pd.DataFrame, n_boot: int) -> None:
    method_files = {
        "Deterministic": "pred_deterministic",
        "MC Dropout": "pred_mc_dropout",
        "Deep Ensemble": "pred_deep_ensemble",
        "Residual Gaussian": "pred_bayesian_laplace",
    }
    rows = []
    for variant in ["raw", "calibrated"]:
        for method, stem in method_files.items():
            frames = []
            for repeat in [1, 2, 3]:
                path = EXP2 / f"repeat{repeat}" / "m4" / f"{stem}_{variant}.csv"
                if not path.exists():
                    continue
                pred = add_patient_info(pd.read_csv(path), labels)
                pred["repeat"] = repeat
                frames.append(pred)
            if not frames:
                continue
            summarize_cluster_result(
                rows,
                {"m": 4, "method": method, "variant": variant},
                frames,
                n_boot=n_boot,
                seed=41000 + len(rows),
            )
    pd.DataFrame(rows).to_csv(OUT / "experiment2_m4_uq_patient_cluster_bootstrap.csv", index=False)


def other_subgroup_composition(labels: pd.DataFrame) -> None:
    df = load_deeplesion_len5_long(relative_log=True)
    ids = sorted(df["trajectory_id"].astype(str).unique().tolist())
    cfg = ExperimentConfig()
    split = split_trajectory_ids_by_patient(
        df,
        ids,
        seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )
    rows = []
    for split_name, split_ids in [("full cohort", ids), ("test", split.test)]:
        sub = labels[(labels["trajectory_id"].isin(split_ids)) & (labels["subgroup"] == "other")].copy()
        for keys, group in sub.groupby(["body_site_primary", "lesion_type_primary"], dropna=False):
            rows.append(
                {
                    "split": split_name,
                    "body_site_primary": keys[0],
                    "lesion_type_primary": keys[1],
                    "N_trajectories": len(group),
                    "N_patients": group["patient_id"].nunique(),
                }
            )
    pd.DataFrame(rows).sort_values(["split", "N_trajectories"], ascending=[True, False]).to_csv(
        OUT / "other_subgroup_composition.csv", index=False
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = patient_map_and_labels()
    n_boot = 1000
    experiment1_neural_and_subgroups(labels, n_boot=n_boot)
    experiment1_paired_differences(labels, n_boot=n_boot)
    experiment2_m4_uq_cluster_bootstrap(labels, n_boot=n_boot)
    other_subgroup_composition(labels)
    print(f"Wrote patient-level cluster bootstrap outputs to {OUT}")
    print(pd.read_csv(OUT / "experiment1_neural_patient_cluster_bootstrap.csv")[["m", "method", "RMSE", "MAE", "PICP", "MPIW"]].to_string(index=False))
    print()
    print(pd.read_csv(OUT / "experiment1_paired_rmse_difference_cluster_bootstrap.csv").query("m == 4").to_string(index=False))


if __name__ == "__main__":
    main()
