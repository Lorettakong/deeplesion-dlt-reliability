from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge

from nlstt_adaptive_uq_experiments.config import ExperimentConfig
from nlstt_adaptive_uq_experiments.data import (
    load_deeplesion_len5_long,
    make_deeplesion_prediction_task,
    split_trajectory_ids_by_patient,
)
from nlstt_adaptive_uq_experiments.mechanistic_baselines import (
    _gompertz_fit_predict,
    _logistic_fit_predict,
    _log_linear_fit_predict,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_nlstt_adaptive_uq_paper" / "experiment1_traditional_baselines"
EXP1 = ROOT / "outputs_nlstt_adaptive_uq_paper" / "experiment1_followup_density"


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    rmse_vals = []
    mae_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rmse_vals.append(rmse(y[idx], pred[idx]))
        mae_vals.append(mae(y[idx], pred[idx]))
    return (
        float(np.quantile(rmse_vals, 0.025)),
        float(np.quantile(rmse_vals, 0.975)),
        float(np.quantile(mae_vals, 0.025)),
        float(np.quantile(mae_vals, 0.975)),
    )


def format_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.4f} ({lo:.4f}-{hi:.4f})"


def task_features(task: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in task.iterrows():
        t = np.asarray(row["t_obs"], dtype=float)
        y = np.asarray(row["logv_obs"], dtype=float)
        target_t = float(row["t_target"])
        if len(y) >= 2:
            local_slope = (y[-1] - y[-2]) / max(t[-1] - t[-2], 1e-6)
            global_slope = (y[-1] - y[0]) / max(t[-1] - t[0], 1e-6)
        else:
            local_slope = 0.0
            global_slope = 0.0
        rows.append(
            {
                "trajectory_id": str(row["trajectory_id"]),
                "last_y": float(y[-1]),
                "mean_y": float(np.mean(y)),
                "std_y": float(np.std(y)),
                "local_slope": float(local_slope),
                "global_slope": float(global_slope),
                "target_dt": float(target_t - t[-1]),
                "m": int(row["m"]),
                "logv_target": float(row["logv_target"]),
            }
        )
    return pd.DataFrame(rows)


def tune_ridge(train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    x_train = task_features(train_df)
    x_val = task_features(val_df)
    feature_cols = ["last_y", "mean_y", "std_y", "local_slope", "global_slope", "target_dt", "m"]
    best_alpha = 1.0
    best_rmse = float("inf")
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(x_train[feature_cols], x_train["logv_target"])
        pred = model.predict(x_val[feature_cols])
        score = rmse(x_val["logv_target"].to_numpy(float), pred)
        if score < best_rmse:
            best_rmse = score
            best_alpha = alpha
    return best_alpha


def predict_ridge(train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float) -> np.ndarray:
    feature_cols = ["last_y", "mean_y", "std_y", "local_slope", "global_slope", "target_dt", "m"]
    x_train = task_features(train_df)
    x_test = task_features(test_df)
    model = Ridge(alpha=alpha)
    model.fit(x_train[feature_cols], x_train["logv_target"])
    return model.predict(x_test[feature_cols]).astype(float)


def gp_predict_one(t_obs: np.ndarray, y_obs: np.ndarray, t_target: float, length_scale: float, noise: float) -> float:
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds="fixed") + WhiteKernel(
        noise_level=noise, noise_level_bounds="fixed"
    )
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)
    gp.fit(t_obs.reshape(-1, 1), y_obs)
    return float(gp.predict(np.asarray([[t_target]], dtype=float))[0])


def tune_gp(val_df: pd.DataFrame) -> tuple[float, float]:
    best = (1.0, 0.01)
    best_rmse = float("inf")
    y_true = val_df["logv_target"].to_numpy(float)
    for length_scale in [0.5, 1.0, 2.0, 4.0]:
        for noise in [1e-4, 1e-3, 1e-2, 5e-2]:
            preds = []
            for _, row in val_df.iterrows():
                preds.append(
                    gp_predict_one(
                        np.asarray(row["t_obs"], dtype=float),
                        np.asarray(row["logv_obs"], dtype=float),
                        float(row["t_target"]),
                        length_scale=length_scale,
                        noise=noise,
                    )
                )
            score = rmse(y_true, np.asarray(preds, dtype=float))
            if score < best_rmse:
                best_rmse = score
                best = (length_scale, noise)
    return best


def predict_gp(task_df: pd.DataFrame, length_scale: float, noise: float) -> np.ndarray:
    preds = []
    for _, row in task_df.iterrows():
        preds.append(
            gp_predict_one(
                np.asarray(row["t_obs"], dtype=float),
                np.asarray(row["logv_obs"], dtype=float),
                float(row["t_target"]),
                length_scale=length_scale,
                noise=noise,
            )
        )
    return np.asarray(preds, dtype=float)


def predict_curve(task_df: pd.DataFrame, kind: str) -> np.ndarray:
    preds = []
    for _, row in task_df.iterrows():
        t = np.asarray(row["t_obs"], dtype=float)
        y = np.asarray(row["logv_obs"], dtype=float)
        t_target = float(row["t_target"])
        if kind == "least_squares_linear":
            preds.append(_log_linear_fit_predict(t, y, t_target))
        elif kind == "recent_two_linear":
            preds.append(_log_linear_fit_predict(t[-2:], y[-2:], t_target))
        elif kind == "gompertz":
            preds.append(_gompertz_fit_predict(t, y, t_target))
        elif kind == "logistic":
            preds.append(_logistic_fit_predict(t, y, t_target))
        else:
            raise ValueError(kind)
    return np.asarray(preds, dtype=float)


def summarize_method(
    records: list[dict],
    pred_rows: list[pd.DataFrame],
    m: int,
    method: str,
    applicability: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
    seed: int,
    selected_hyperparameter: str = "",
) -> None:
    r = rmse(y_true, y_pred)
    a = mae(y_true, y_pred)
    r_lo, r_hi, a_lo, a_hi = bootstrap_ci(y_true, y_pred, seed=seed)
    records.append(
        {
            "m": m,
            "method": method,
            "N_test": len(y_true),
            "rmse": r,
            "rmse_ci95_low": r_lo,
            "rmse_ci95_high": r_hi,
            "mae": a,
            "mae_ci95_low": a_lo,
            "mae_ci95_high": a_hi,
            "RMSE_95CI": format_ci(r, r_lo, r_hi),
            "MAE_95CI": format_ci(a, a_lo, a_hi),
            "applicability": applicability,
            "selected_hyperparameter": selected_hyperparameter,
        }
    )
    pred_rows.append(
        pd.DataFrame(
            {
                "trajectory_id": test_df["trajectory_id"].astype(str),
                "m": m,
                "method": method,
                "logv_target": y_true,
                "logv_mean": y_pred,
            }
        )
    )


def load_neural_reference() -> pd.DataFrame:
    path = EXP1 / "experiment1_pooled_validation_selected_rmse.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    rows = []
    for _, row in table.iterrows():
        rows.append(
            {
                "m": int(row["m"]),
                "method": "Validation-selected neural/UQ model",
                "N_test": int(row["N_test"]),
                "rmse": float(row["rmse"]),
                "rmse_ci95_low": np.nan,
                "rmse_ci95_high": np.nan,
                "mae": np.nan,
                "mae_ci95_low": np.nan,
                "mae_ci95_high": np.nan,
                "RMSE_95CI": row["RMSE"],
                "MAE_95CI": "",
                "applicability": "Main neural benchmark selected by validation-set RMSE.",
                "selected_hyperparameter": str(row.get("selected_method", "")),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    methods = [
        "Training-set mean target",
        "Training-set median target",
        "LOCF",
        "Ridge trajectory regression",
        "Recent-two linear extrapolation",
        "Gaussian process regression",
        "Gompertz curve fit",
        "Logistic curve fit",
        "Validation-selected neural/UQ model",
    ]
    for method in methods:
        sub = summary[summary["method"] == method].sort_values("m")
        if sub.empty:
            continue
        ax.plot(sub["m"], sub["rmse"], marker="o", label=method)
    ax.set_xlabel("Observed CT visits (m)")
    ax.set_ylabel("Test RMSE of relative log-volume")
    ax.set_title("Traditional longitudinal baselines vs validation-selected neural/UQ result")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_experiment1_traditional_baselines.png", dpi=300)
    fig.savefig(OUT / "fig_experiment1_traditional_baselines.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig()
    df = load_deeplesion_len5_long(relative_log=True)
    ids = sorted(df["trajectory_id"].astype(str).unique().tolist())
    split = split_trajectory_ids_by_patient(
        df,
        ids,
        seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )

    records: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    for m in [1, 2, 3, 4]:
        train_df = make_deeplesion_prediction_task(df, split.train, m=m)
        val_df = make_deeplesion_prediction_task(df, split.val, m=m)
        test_df = make_deeplesion_prediction_task(df, split.test, m=m)
        y_true = test_df["logv_target"].to_numpy(float)

        mean_value = float(train_df["logv_target"].mean())
        median_value = float(train_df["logv_target"].median())
        summarize_method(
            records,
            pred_rows,
            m,
            "Training-set mean target",
            "Population baseline; uses no individualized longitudinal change.",
            y_true,
            np.full_like(y_true, mean_value, dtype=float),
            test_df,
            seed=11000 + m,
            selected_hyperparameter=f"mean={mean_value:.6f}",
        )
        summarize_method(
            records,
            pred_rows,
            m,
            "Training-set median target",
            "Population baseline; robust cohort-prior prediction.",
            y_true,
            np.full_like(y_true, median_value, dtype=float),
            test_df,
            seed=12000 + m,
            selected_hyperparameter=f"median={median_value:.6f}",
        )
        locf_pred = np.asarray([float(obs[-1]) for obs in test_df["logv_obs"]], dtype=float)
        summarize_method(
            records,
            pred_rows,
            m,
            "LOCF",
            "Last observation carried forward; for m=1 this equals the zero-change baseline.",
            y_true,
            locf_pred,
            test_df,
            seed=13000 + m,
        )

        ridge_alpha = tune_ridge(train_df, val_df)
        summarize_method(
            records,
            pred_rows,
            m,
            "Ridge trajectory regression",
            "Validation-tuned linear regression on simple trajectory features.",
            y_true,
            predict_ridge(train_df, test_df, alpha=ridge_alpha),
            test_df,
            seed=14000 + m,
            selected_hyperparameter=f"alpha={ridge_alpha:g}",
        )

        if m >= 2:
            summarize_method(
                records,
                pred_rows,
                m,
                "Least-squares linear extrapolation",
                "Fits a linear trend to all observed points and extrapolates to T4.",
                y_true,
                predict_curve(test_df, "least_squares_linear"),
                test_df,
                seed=15000 + m,
            )
            summarize_method(
                records,
                pred_rows,
                m,
                "Recent-two linear extrapolation",
                "Uses the two most recent observed points to estimate the slope.",
                y_true,
                predict_curve(test_df, "recent_two_linear"),
                test_df,
                seed=16000 + m,
            )
            gp_length, gp_noise = tune_gp(val_df)
            summarize_method(
                records,
                pred_rows,
                m,
                "Gaussian process regression",
                "Per-trajectory RBF Gaussian process; length scale and noise selected on validation set.",
                y_true,
                predict_gp(test_df, gp_length, gp_noise),
                test_df,
                seed=17000 + m,
                selected_hyperparameter=f"length_scale={gp_length:g}; noise={gp_noise:g}",
            )
        if m >= 3:
            summarize_method(
                records,
                pred_rows,
                m,
                "Gompertz curve fit",
                "Parametric growth-curve baseline; only evaluated when at least three visits are observed.",
                y_true,
                predict_curve(test_df, "gompertz"),
                test_df,
                seed=18000 + m,
            )
            summarize_method(
                records,
                pred_rows,
                m,
                "Logistic curve fit",
                "Parametric growth-curve baseline; only evaluated when at least three visits are observed.",
                y_true,
                predict_curve(test_df, "logistic"),
                test_df,
                seed=19000 + m,
            )

    summary = pd.DataFrame(records)
    neural = load_neural_reference()
    if not neural.empty:
        summary_with_neural = pd.concat([summary, neural], ignore_index=True)
    else:
        summary_with_neural = summary.copy()
    predictions = pd.concat(pred_rows, ignore_index=True)

    summary.to_csv(OUT / "experiment1_traditional_baseline_summary.csv", index=False)
    summary_with_neural.to_csv(OUT / "experiment1_traditional_baseline_summary_with_neural_reference.csv", index=False)
    predictions.to_csv(OUT / "experiment1_traditional_baseline_predictions.csv", index=False)

    wide = summary.pivot(index="method", columns="m", values="RMSE_95CI").reset_index()
    wide.columns = ["Method"] + [f"m={c} RMSE (95% bootstrap CI)" for c in wide.columns[1:]]
    wide.to_csv(OUT / "table1c_traditional_baseline_rmse.csv", index=False)

    plot_summary(summary_with_neural)
    print(f"Wrote traditional baseline outputs to {OUT}")
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
