from __future__ import annotations

import argparse
from pathlib import Path

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


OUT = Path("outputs_nlstt_adaptive_uq_paper/measurement_proxy_sensitivity")
PROXIES = ["ellipsoid_volume", "recist_area", "long_axis"]


def run(repeats: int = 10, epochs: int = 300, m: int = 4) -> Path:
    """Sensitivity of accuracy/physics conclusions to the RECIST-derived target scale.

    The patient split, target visit, model architecture, and seeds are shared across
    proxies.  Only the measurement transformed to relative log scale changes.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig()
    train_cfg = TrainConfig(epochs=epochs, batch_size=128, hidden_dim=64, dropout=0.10)
    rows = []
    predictions = []
    for proxy in PROXIES:
        df = load_deeplesion_len5_long(relative_log=True, measurement_proxy=proxy)
        ids = sorted(df["trajectory_id"].astype(str).unique())
        split = split_trajectory_ids_by_patient(
            df, ids, seed=cfg.cohort.seed + 205,
            test_fraction=cfg.cohort.test_fraction,
            val_fraction=cfg.cohort.val_fraction,
        )
        reference = estimate_fixed_gompertz_reference(df, split.train)
        train_df = make_deeplesion_prediction_task(df, split.train, m=m)
        test_df = make_deeplesion_prediction_task(df, split.test, m=m)
        for repeat in range(repeats):
            for lam in [0.0, 1.0]:
                method = "no_physics" if lam == 0 else "fixed_pinn"
                run_cfg = train_cfg.__class__(**{**train_cfg.__dict__, "fixed_lambda_phys": lam})
                seed = 160000 + repeat * 1000 + int(lam * 100) + PROXIES.index(proxy)
                trained = train_single_model(
                    train_df, run_cfg, method=method, seed=seed,
                    gompertz_reference=reference,
                )
                pred = predict_deterministic(trained, test_df)
                pred["measurement_proxy"] = proxy
                pred["repeat"] = repeat + 1
                pred["lambda_phys"] = lam
                predictions.append(pred)
                rows.append(
                    {
                        "measurement_proxy": proxy,
                        "repeat": repeat + 1,
                        "m": m,
                        "lambda_phys": lam,
                        "seed": seed,
                        "gompertz_alpha_training_only": reference.alpha,
                        "gompertz_log_k_training_only": reference.log_k,
                        **summarize_predictions(pred),
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "measurement_proxy_seed_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "measurement_proxy_predictions.csv", index=False)
    summary = (
        metrics.groupby(["measurement_proxy", "m", "lambda_phys"], as_index=False)
        .agg(
            n_seeds=("repeat", "nunique"),
            rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
            residual_mean=("physics_residual_abs", "mean"),
            residual_sd=("physics_residual_abs", "std"),
        )
    )
    summary.to_csv(OUT / "measurement_proxy_sensitivity_summary.csv", index=False)
    paired = []
    for proxy, group in metrics.groupby("measurement_proxy"):
        pivot = group.pivot(index="repeat", columns="lambda_phys", values=["rmse", "physics_residual_abs"])
        paired.append(
            {
                "measurement_proxy": proxy,
                "delta_rmse_lambda1_minus_0_mean": float((pivot["rmse"][1.0] - pivot["rmse"][0.0]).mean()),
                "residual_ratio_lambda1_over_0_mean": float((pivot["physics_residual_abs"][1.0] / np.clip(pivot["physics_residual_abs"][0.0], 1e-12, None)).mean()),
            }
        )
    pd.DataFrame(paired).to_csv(OUT / "measurement_proxy_paired_conclusion_check.csv", index=False)
    return OUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--m", type=int, default=4)
    args = parser.parse_args()
    print(run(repeats=args.repeats, epochs=args.epochs, m=args.m))
