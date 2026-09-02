from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DEEPLESION_TRAJ_LABELS_CSV,
    DEEPLESION_TRAJ_LEN5_CSV,
    NLSTT_ALL_ELIGIBLE_SUMMARY_CSV,
    NLSTT_LONG_CSV,
)


GROWTH_CLASSES = ("decreasing", "stable", "slow_growth", "rapid_growth")


@dataclass(frozen=True)
class SplitIds:
    train: list[str]
    val: list[str]
    test: list[str]


def _summary_to_long(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        for t, col in [(0.0, "v0_cm3"), (1.0, "v1_cm3"), (2.0, "v2_cm3")]:
            rows.append(
                {
                    "patient_id": row["patient_id"],
                    "trajectory_id": str(row["trajectory_id"]),
                    "growth_class": row["growth_class"],
                    "t_rel": t,
                    "V_obs_cm3": float(row[col]),
                }
            )
    return pd.DataFrame(rows)


def load_nlstt_long(path: Path = NLSTT_ALL_ELIGIBLE_SUMMARY_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"v0_cm3", "v1_cm3", "v2_cm3"}.issubset(df.columns):
        df = _summary_to_long(df)
    required = {"patient_id", "trajectory_id", "t_rel", "V_obs_cm3", "growth_class"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"NLSTt file is missing columns: {sorted(missing)}")
    df = df.copy()
    df["trajectory_id"] = df["trajectory_id"].astype(str)
    df["logV"] = np.log(np.clip(df["V_obs_cm3"].astype(float), 1e-8, None))
    return df.sort_values(["trajectory_id", "t_rel"]).reset_index(drop=True)


def balanced_sample_trajectories(df: pd.DataFrame, n: int, seed: int) -> list[str]:
    summary = df.groupby("trajectory_id", as_index=False).agg(growth_class=("growth_class", "first"))
    rng = np.random.default_rng(seed)
    per_class = n // len(GROWTH_CLASSES)
    selected: list[str] = []
    leftovers: list[str] = []
    for cls in GROWTH_CLASSES:
        ids = summary.loc[summary["growth_class"] == cls, "trajectory_id"].to_numpy()
        rng.shuffle(ids)
        selected.extend(ids[:per_class].tolist())
        leftovers.extend(ids[per_class:].tolist())
    if len(selected) < n:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])
    return sorted(selected[:n])


def split_trajectory_ids(
    ids: list[str],
    seed: int,
    test_fraction: float = 0.20,
    val_fraction: float = 0.15,
) -> SplitIds:
    rng = np.random.default_rng(seed)
    ids_arr = np.array(ids, dtype=object)
    rng.shuffle(ids_arr)
    n = len(ids_arr)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test = ids_arr[:n_test].tolist()
    val = ids_arr[n_test : n_test + n_val].tolist()
    train = ids_arr[n_test + n_val :].tolist()
    return SplitIds(train=train, val=val, test=test)


def split_trajectory_ids_by_patient(
    df: pd.DataFrame,
    ids: list[str],
    seed: int,
    test_fraction: float = 0.20,
    val_fraction: float = 0.15,
) -> SplitIds:
    """Split trajectory ids through patient ids to avoid patient leakage."""
    summary = (
        df[df["trajectory_id"].isin(ids)]
        .groupby("trajectory_id", as_index=False)
        .agg(patient_id=("patient_id", "first"))
    )
    rng = np.random.default_rng(seed)
    patients = summary["patient_id"].astype(str).drop_duplicates().to_numpy(dtype=object)
    rng.shuffle(patients)
    n = len(patients)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test_patients = set(patients[:n_test].tolist())
    val_patients = set(patients[n_test : n_test + n_val].tolist())

    test = summary.loc[summary["patient_id"].astype(str).isin(test_patients), "trajectory_id"].astype(str).tolist()
    val = summary.loc[summary["patient_id"].astype(str).isin(val_patients), "trajectory_id"].astype(str).tolist()
    train = summary.loc[
        ~summary["patient_id"].astype(str).isin(test_patients | val_patients),
        "trajectory_id",
    ].astype(str).tolist()
    return SplitIds(train=sorted(train), val=sorted(val), test=sorted(test))


def patient_group_kfold_ids(
    df: pd.DataFrame,
    ids: list[str],
    n_splits: int = 5,
    seed: int = 0,
) -> list[tuple[list[str], list[str]]]:
    """Patient-grouped folds for development/model selection only.

    Every patient is assigned to exactly one validation fold, so trajectories
    from the same patient can never occur on both sides of a fold.
    """
    summary = (
        df[df["trajectory_id"].astype(str).isin([str(x) for x in ids])]
        .groupby("trajectory_id", as_index=False)
        .agg(patient_id=("patient_id", "first"))
    )
    patients = summary["patient_id"].astype(str).drop_duplicates().to_numpy(dtype=object)
    if len(patients) < n_splits:
        raise ValueError(f"Need at least {n_splits} development patients, found {len(patients)}.")
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    patient_folds = np.array_split(patients, n_splits)
    folds: list[tuple[list[str], list[str]]] = []
    for fold_patients in patient_folds:
        val_patients = set(str(x) for x in fold_patients.tolist())
        is_val = summary["patient_id"].astype(str).isin(val_patients)
        train_ids = summary.loc[~is_val, "trajectory_id"].astype(str).tolist()
        val_ids = summary.loc[is_val, "trajectory_id"].astype(str).tolist()
        folds.append((sorted(train_ids), sorted(val_ids)))
    return folds


def make_prediction_task(df: pd.DataFrame, ids: list[str], m: int) -> pd.DataFrame:
    """Create sparse prediction rows.

    Real NLSTt supports m=1,2,3 only:
    - m=1: use T0 to predict T2
    - m=2: use T0,T1 to predict T2
    - m=3: use all three points for fitting/reconstruction diagnostics
    """
    if m not in {1, 2, 3}:
        raise ValueError("Real NLSTt tasks only support m in {1, 2, 3}. Use semi_synthetic_followups for m=5/8.")
    sub = df[df["trajectory_id"].isin(ids)].copy()
    rows = []
    for tid, group in sub.groupby("trajectory_id"):
        group = group.sort_values("t_rel")
        times = group["t_rel"].to_numpy(dtype=float)
        vols = group["V_obs_cm3"].to_numpy(dtype=float)
        logv = group["logV"].to_numpy(dtype=float)
        if len(group) != 3:
            continue
        obs_idx = np.arange(m)
        target_idx = 2
        rows.append(
            {
                "trajectory_id": tid,
                "growth_class": group["growth_class"].iloc[0],
                "m": m,
                "t_obs": times[obs_idx].tolist(),
                "logv_obs": logv[obs_idx].tolist(),
                "v_obs": vols[obs_idx].tolist(),
                "t_target": float(times[target_idx]),
                "logv_target": float(logv[target_idx]),
                "v_target": float(vols[target_idx]),
            }
        )
    return pd.DataFrame(rows)


def fit_log_linear_growth(t: np.ndarray, logv: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(t, logv, 1)
    return float(intercept), float(slope)


def _fit_gompertz_grid(t: np.ndarray, logv: np.ndarray) -> tuple[float, float, float]:
    """Fit y(t)=logK-(logK-y0)*exp(-alpha*t) with a small robust grid search."""
    y0 = float(logv[0])
    y_max = float(np.max(logv))
    y_min = float(np.min(logv))
    best = (float("inf"), 0.1, y_max + 0.5)
    logk_grid = np.linspace(y_max + 0.05, y_max + 2.0, 60)
    alpha_grid = np.linspace(0.01, 2.0, 80)
    for log_k in logk_grid:
        base = log_k - y0
        for alpha in alpha_grid:
            pred = log_k - base * np.exp(-alpha * t)
            loss = float(np.mean((pred - logv) ** 2))
            if loss < best[0]:
                best = (loss, float(alpha), float(log_k))
    _, alpha, log_k = best
    return y0, alpha, log_k


def _gompertz_logv(t: np.ndarray, y0: float, alpha: float, log_k: float) -> np.ndarray:
    return log_k - (log_k - y0) * np.exp(-alpha * t)


def _fit_logistic_grid(t: np.ndarray, v: np.ndarray) -> tuple[float, float, float]:
    """Fit V(t)=K/(1+A*exp(-r*t)) using a small robust grid search."""
    v0 = float(max(v[0], 1e-6))
    vmax = float(np.max(v))
    best = (float("inf"), 0.1, vmax * 2.0)
    k_grid = np.linspace(max(vmax * 1.05, v0 * 1.1), max(vmax * 8.0, v0 * 2.0), 60)
    r_grid = np.linspace(0.01, 3.0, 90)
    for k_cap in k_grid:
        a = k_cap / v0 - 1.0
        if a <= 0:
            continue
        for r in r_grid:
            pred = k_cap / (1.0 + a * np.exp(-r * t))
            loss = float(np.mean((np.log(np.clip(pred, 1e-8, None)) - np.log(np.clip(v, 1e-8, None))) ** 2))
            if loss < best[0]:
                best = (loss, float(r), float(k_cap))
    _, r, k_cap = best
    return v0, r, k_cap


def _logistic_v(t: np.ndarray, v0: float, r: float, k_cap: float) -> np.ndarray:
    a = k_cap / max(v0, 1e-8) - 1.0
    return k_cap / (1.0 + a * np.exp(-r * t))


def generate_dense_followup_dataset(
    df: pd.DataFrame,
    ids: list[str],
    total_nodes: int,
    generator: str,
    seed: int,
    noise_sd: float = 0.05,
) -> pd.DataFrame:
    """Generate L-node semi-synthetic trajectories anchored to real T0,T1,T2 records."""
    if total_nodes not in {5, 8}:
        raise ValueError("total_nodes should be 5 or 8 for this experiment.")
    if generator not in {"gompertz", "logistic"}:
        raise ValueError("generator should be 'gompertz' or 'logistic'.")
    rng = np.random.default_rng(seed)
    sub = df[df["trajectory_id"].isin(ids)].copy()
    rows = []
    dense_t = np.linspace(0.0, 2.0, total_nodes)
    for tid, group in sub.groupby("trajectory_id"):
        group = group.sort_values("t_rel")
        if len(group) != 3:
            continue
        t = group["t_rel"].to_numpy(dtype=float)
        v = group["V_obs_cm3"].to_numpy(dtype=float)
        logv = np.log(np.clip(v, 1e-8, None))
        if generator == "gompertz":
            y0, alpha, log_k = _fit_gompertz_grid(t, logv)
            dense_logv = _gompertz_logv(dense_t, y0, alpha, log_k)
        else:
            v0, r, k_cap = _fit_logistic_grid(t, v)
            dense_logv = np.log(np.clip(_logistic_v(dense_t, v0, r, k_cap), 1e-8, None))
        dense_logv = dense_logv + rng.normal(0.0, noise_sd, size=total_nodes)
        for j, (tt, yy) in enumerate(zip(dense_t, dense_logv)):
            rows.append(
                {
                    "patient_id": group["patient_id"].iloc[0],
                    "trajectory_id": str(tid),
                    "growth_class": group["growth_class"].iloc[0],
                    "node_index": j,
                    "t_rel": float(tt),
                    "V_obs_cm3": float(np.exp(yy)),
                    "logV": float(yy),
                    "total_nodes": total_nodes,
                    "generator": generator,
                }
            )
    return pd.DataFrame(rows).sort_values(["trajectory_id", "t_rel"]).reset_index(drop=True)


def make_dense_prediction_task(dense_df: pd.DataFrame, ids: list[str], k: int) -> pd.DataFrame:
    """Use first k dense nodes to predict the final dense node."""
    rows = []
    sub = dense_df[dense_df["trajectory_id"].isin(ids)].copy()
    for tid, group in sub.groupby("trajectory_id"):
        group = group.sort_values("t_rel")
        if len(group) <= k:
            continue
        obs = group.iloc[:k]
        target = group.iloc[-1]
        rows.append(
            {
                "trajectory_id": str(tid),
                "growth_class": group["growth_class"].iloc[0],
                "m": k,
                "total_nodes": int(group["total_nodes"].iloc[0]),
                "generator": group["generator"].iloc[0],
                "t_obs": obs["t_rel"].to_numpy(float).tolist(),
                "logv_obs": obs["logV"].to_numpy(float).tolist(),
                "v_obs": obs["V_obs_cm3"].to_numpy(float).tolist(),
                "t_target": float(target["t_rel"]),
                "logv_target": float(target["logV"]),
                "v_target": float(target["V_obs_cm3"]),
            }
        )
    return pd.DataFrame(rows)


def _growth_class_from_endpoints(v0: float, v_final: float) -> str:
    rel = (v_final - v0) / max(abs(v0), 1e-8)
    if rel < -0.20:
        return "decreasing"
    if rel < 0.20:
        return "stable"
    if rel < 1.00:
        return "slow_growth"
    return "rapid_growth"


def load_deeplesion_len5_long(
    path: Path = DEEPLESION_TRAJ_LEN5_CSV,
    labels_path: Path | None = DEEPLESION_TRAJ_LABELS_CSV,
    relative_log: bool = False,
    cohort_mode: str = "current",
    measurement_proxy: str = "ellipsoid_volume",
) -> pd.DataFrame:
    """Load DLT-derived same-lesion trajectories and keep the first five real follow-ups.

    The DLT annotations contain RECIST long/short diameters instead of manual
    segmentations. We therefore model an ellipsoid volume proxy:
    V = pi / 6 * long_axis * short_axis^2, converted from mm^3 to cm^3.
    """
    path = Path(path)
    if cohort_mode == "strict_unambiguous" and not path.stem.endswith("_strict"):
        path = path.with_name(f"{path.stem}_strict{path.suffix}")
    if cohort_mode == "strict_unambiguous" and not path.exists():
        raise FileNotFoundError(
            f"Strict cohort file not found: {path}. Generate it first with "
            "`python prepare_deeplesion_longitudinal.py --ambiguity-policy exclude`."
        )
    df = pd.read_csv(path)
    required = {
        "trajectory_id",
        "followup_index",
        "patient_id",
        "study_id",
        "long_axis_mm",
        "short_axis_mm",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DeepLesion trajectory file is missing columns: {sorted(missing)}")

    if cohort_mode not in {"current", "strict_unambiguous"}:
        raise ValueError("cohort_mode must be 'current' or 'strict_unambiguous'.")
    df = df.copy()
    df["trajectory_id"] = df["trajectory_id"].astype(str)
    df["followup_index"] = df["followup_index"].astype(int)
    df = df.sort_values(["trajectory_id", "followup_index"])
    df = df[df["followup_index"] < 5].copy()

    if cohort_mode == "strict_unambiguous" and "ambiguity_policy" in df.columns:
        if not df["ambiguity_policy"].astype(str).eq("exclude").all():
            raise ValueError("Strict cohort file contains rows not constructed with ambiguity_policy='exclude'.")
    df["cohort_construction"] = cohort_mode

    long_axis = df["long_axis_mm"].astype(float)
    short_axis = df["short_axis_mm"].astype(float)
    if measurement_proxy == "ellipsoid_volume":
        measurement = (math.pi / 6.0) * long_axis * short_axis**2 / 1000.0
        proxy_label = "pi_over_6_times_long_times_short_squared_cm3"
    elif measurement_proxy == "recist_area":
        measurement = long_axis * short_axis
        proxy_label = "long_times_short_mm2"
    elif measurement_proxy == "long_axis":
        measurement = long_axis
        proxy_label = "recist_long_axis_mm"
    else:
        raise ValueError("measurement_proxy must be ellipsoid_volume, recist_area, or long_axis.")
    df["V_obs_cm3"] = np.clip(measurement, 1e-8, None)
    df["measurement_proxy"] = measurement_proxy
    df["measurement_proxy_definition"] = proxy_label
    df["logV"] = np.log(df["V_obs_cm3"])
    df["raw_logV"] = df["logV"]
    df["t_rel"] = df["followup_index"].astype(float)
    df["node_index"] = df["followup_index"].astype(int)
    df["total_nodes"] = 5

    complete_ids = df.groupby("trajectory_id").size()
    complete_ids = complete_ids[complete_ids == 5].index
    df = df[df["trajectory_id"].isin(complete_ids)].copy()

    classes = {}
    for tid, group in df.groupby("trajectory_id"):
        group = group.sort_values("t_rel")
        classes[tid] = _growth_class_from_endpoints(
            float(group["V_obs_cm3"].iloc[0]),
            float(group["V_obs_cm3"].iloc[-1]),
        )
    df["growth_class"] = df["trajectory_id"].map(classes)
    if labels_path is not None and Path(labels_path).exists():
        labels = pd.read_csv(labels_path)
        keep_cols = [
            c
            for c in [
                "trajectory_id",
                "body_region_group",
                "body_site_primary",
                "lesion_type_primary",
                "lung_or_chest_lung_terms",
            ]
            if c in labels.columns
        ]
        if len(keep_cols) > 1:
            df = df.merge(labels[keep_cols], on="trajectory_id", how="left")
    if "body_region_group" not in df.columns:
        df["body_region_group"] = "unknown"

    if relative_log:
        baselines = df.sort_values("t_rel").groupby("trajectory_id")["raw_logV"].first()
        df["logV_baseline"] = df["trajectory_id"].map(baselines)
        df["logV"] = df["raw_logV"] - df["logV_baseline"]
        df["V_obs_cm3_raw"] = df["V_obs_cm3"]
        df["V_obs_cm3"] = np.exp(df["logV"])
        df["target_transform"] = f"relative_log_{measurement_proxy}_over_baseline"
    else:
        df["logV_baseline"] = 0.0
        df["V_obs_cm3_raw"] = df["V_obs_cm3"]
        df["target_transform"] = "raw_log_volume"
    return df.sort_values(["trajectory_id", "t_rel"]).reset_index(drop=True)


def make_deeplesion_prediction_task(df: pd.DataFrame, ids: list[str], m: int) -> pd.DataFrame:
    """Use the most recent m pre-target visits to predict the fixed T4 target.

    For a five-visit trajectory (T0, ..., T4), the forecast horizon is fixed at
    T4 and the observed history expands backwards from the last available
    pre-target visit: m=1 uses T3, m=2 uses (T2, T3), m=3 uses
    (T1, T2, T3), and m=4 uses (T0, T1, T2, T3).  This isolates the
    incremental information supplied by a longer recent history while keeping
    both the target and the final observed visit fixed.
    """
    if m not in {1, 2, 3, 4}:
        raise ValueError("DeepLesion len5 tasks support m in {1, 2, 3, 4}.")
    rows = []
    sub = df[df["trajectory_id"].isin(ids)].copy()
    for tid, group in sub.groupby("trajectory_id"):
        group = group.sort_values("t_rel")
        if len(group) < 5:
            continue
        obs = group.iloc[4 - m : 4]
        target = group.iloc[4]
        rows.append(
            {
                "trajectory_id": str(tid),
                "patient_id": str(group["patient_id"].iloc[0]),
                "growth_class": group["growth_class"].iloc[0],
                "m": m,
                "forecast_mode": "fixed_horizon_recent_history",
                "individualized_history_available": True,
                "total_nodes": 5,
                "t_obs": obs["t_rel"].to_numpy(float).tolist(),
                "logv_obs": obs["logV"].to_numpy(float).tolist(),
                "v_obs": obs["V_obs_cm3"].to_numpy(float).tolist(),
                "t_target": float(target["t_rel"]),
                "logv_target": float(target["logV"]),
                "v_target": float(target["V_obs_cm3"]),
            }
        )
    return pd.DataFrame(rows)


def semi_synthetic_followups(df: pd.DataFrame, ids: list[str], m: int, seed: int, noise_sd: float = 0.06) -> pd.DataFrame:
    """Generate m=5 or m=8 semi-synthetic trajectories from real three-point NLSTt anchors."""
    if m not in {5, 8}:
        raise ValueError("Semi-synthetic follow-up sensitivity is intended for m=5 or m=8.")
    rng = np.random.default_rng(seed)
    sub = df[df["trajectory_id"].isin(ids)].copy()
    rows = []
    dense_t = np.linspace(0.0, 2.0, m)
    for tid, group in sub.groupby("trajectory_id"):
        group = group.sort_values("t_rel")
        if len(group) != 3:
            continue
        t = group["t_rel"].to_numpy(dtype=float)
        logv = group["logV"].to_numpy(dtype=float)
        a, b = fit_log_linear_growth(t, logv)
        synthetic_logv = a + b * dense_t + rng.normal(0.0, noise_sd, size=m)
        synthetic_v = np.exp(synthetic_logv)
        rows.append(
            {
                "trajectory_id": tid,
                "growth_class": group["growth_class"].iloc[0],
                "m": m,
                "t_obs": dense_t[:-1].tolist(),
                "logv_obs": synthetic_logv[:-1].tolist(),
                "v_obs": synthetic_v[:-1].tolist(),
                "t_target": float(dense_t[-1]),
                "logv_target": float(synthetic_logv[-1]),
                "v_target": float(synthetic_v[-1]),
                "generator": "log-linear fit to real NLSTt anchors",
            }
        )
    return pd.DataFrame(rows)


def save_cohort_files(out_dir: Path, seed: int, main_n: int, hmc_n: int, robustness_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_nlstt_long()
    for name, n in [("main500", main_n), ("hmc300", hmc_n), ("robustness1000", robustness_n)]:
        ids = balanced_sample_trajectories(df, n=n, seed=seed + n)
        df[df["trajectory_id"].isin(ids)].to_csv(out_dir / f"cohort_{name}_long.csv", index=False)
        split = split_trajectory_ids(ids, seed=seed + n)
        pd.DataFrame({"trajectory_id": split.train, "split": "train"}).to_csv(out_dir / f"split_{name}_train.csv", index=False)
        pd.DataFrame({"trajectory_id": split.val, "split": "val"}).to_csv(out_dir / f"split_{name}_val.csv", index=False)
        pd.DataFrame({"trajectory_id": split.test, "split": "test"}).to_csv(out_dir / f"split_{name}_test.csv", index=False)
