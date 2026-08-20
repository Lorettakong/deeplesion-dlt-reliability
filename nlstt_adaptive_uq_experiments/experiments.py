from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .data import (
    balanced_sample_trajectories,
    generate_dense_followup_dataset,
    load_deeplesion_len5_long,
    load_nlstt_long,
    make_deeplesion_prediction_task,
    make_dense_prediction_task,
    make_prediction_task,
    save_cohort_files,
    split_trajectory_ids_by_patient,
    split_trajectory_ids,
)
from .metrics import conformalize_existing_intervals, residual_calibrate_intervals, summarize_predictions


def prepare_outputs(cfg: ExperimentConfig) -> None:
    save_cohort_files(
        cfg.out_dir,
        seed=cfg.cohort.seed,
        main_n=cfg.cohort.main_n,
        hmc_n=cfg.cohort.hmc_n,
        robustness_n=cfg.cohort.robustness_n,
    )


def run_smoke_experiment(cfg: ExperimentConfig) -> Path:
    """Small run to verify the code path before launching the real experiment."""
    from .train import train_ensemble, train_single_model
    from .uq_methods import predict_deterministic, predict_ensemble, predict_mc_dropout

    out_dir = cfg.out_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_nlstt_long()
    ids = balanced_sample_trajectories(df, n=80, seed=cfg.cohort.seed)
    split = split_trajectory_ids(ids, seed=cfg.cohort.seed)
    train_df = make_prediction_task(df, split.train, m=2)
    test_df = make_prediction_task(df, split.test, m=2)

    quick_cfg = cfg.train.__class__(**{**cfg.train.__dict__, "epochs": 20, "ensemble_size": 2, "mc_samples": 5})
    fixed = train_single_model(train_df, quick_cfg, method="fixed_pinn", seed=1)
    dropout = train_single_model(train_df, quick_cfg, method="mc_dropout", seed=2)
    ensemble = train_ensemble(train_df, quick_cfg, seed=10)

    preds = {
        "fixed_pinn": predict_deterministic(fixed, test_df),
        "mc_dropout": predict_mc_dropout(dropout, test_df, samples=quick_cfg.mc_samples),
        "deep_ensemble": predict_ensemble(ensemble, test_df),
    }
    rows = []
    for method, frame in preds.items():
        frame.to_csv(out_dir / f"pred_{method}.csv", index=False)
        rows.append({"method": method, **summarize_predictions(frame)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    return out_dir / "metrics.csv"


def run_fair_nlstt300_experiment(cfg: ExperimentConfig, epochs: int | None = None, repeats: int = 1) -> Path:
    """Run a fair same-N same-split NLSTt-300 experiment for implemented methods."""
    from .train import train_ensemble, train_single_model
    from .uq_methods import predict_deterministic, predict_ensemble, predict_mc_dropout

    out_dir = cfg.out_dir / "fair_nlstt300"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_nlstt_long()
    ids = balanced_sample_trajectories(df, n=cfg.cohort.hmc_n, seed=cfg.cohort.seed + cfg.cohort.hmc_n)
    split = split_trajectory_ids(
        ids,
        seed=cfg.cohort.seed + cfg.cohort.hmc_n,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )
    train_cfg = cfg.train if epochs is None else cfg.train.__class__(**{**cfg.train.__dict__, "epochs": epochs})

    rows = []
    for repeat in range(repeats):
        repeat_seed = 10000 * repeat
        repeat_dir = out_dir / f"repeat{repeat + 1}"
        repeat_dir.mkdir(exist_ok=True)
        for m in [1, 2, 3]:
            task_dir = repeat_dir / f"m{m}"
            task_dir.mkdir(exist_ok=True)
            train_df = make_prediction_task(df, split.train, m=m)
            val_df = make_prediction_task(df, split.val, m=m)
            test_df = make_prediction_task(df, split.test, m=m)
            pd.concat(
                [
                    train_df.assign(split="train"),
                    val_df.assign(split="val"),
                    test_df.assign(split="test"),
                ],
                ignore_index=True,
            ).to_csv(task_dir / "task_rows.csv", index=False)

            trained = {
                "no_physics": train_single_model(train_df, train_cfg, method="no_physics", seed=repeat_seed + 100 + m),
                "fixed_pinn": train_single_model(train_df, train_cfg, method="fixed_pinn", seed=repeat_seed + 200 + m),
                "adaptive_pinn": train_single_model(train_df, train_cfg, method="adaptive_pinn", seed=repeat_seed + 300 + m),
                "mc_dropout_pinn": train_single_model(train_df, train_cfg, method="mc_dropout", seed=repeat_seed + 400 + m),
            }
            ensemble = train_ensemble(train_df, train_cfg, seed=repeat_seed + 500 + m)

            preds = {
                "no_physics": predict_deterministic(trained["no_physics"], test_df),
                "fixed_pinn": predict_deterministic(trained["fixed_pinn"], test_df),
                "adaptive_pinn": predict_mc_dropout(trained["adaptive_pinn"], test_df, samples=train_cfg.mc_samples),
                "mc_dropout_pinn": predict_mc_dropout(trained["mc_dropout_pinn"], test_df, samples=train_cfg.mc_samples),
                "deep_ensemble_pinn": predict_ensemble(ensemble, test_df),
            }
            for method, frame in preds.items():
                frame.to_csv(task_dir / f"pred_{method}.csv", index=False)
                rows.append({"repeat": repeat + 1, "m": m, "method": method, **summarize_predictions(frame)})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = (
        metrics.groupby(["m", "method"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            picp_mean=("picp", "mean"),
            picp_std=("picp", "std"),
            mpiw_mean=("mpiw", "mean"),
            mpiw_std=("mpiw", "std"),
        )
    )
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    return out_dir / "metrics_summary.csv"


def _summarize_repeated_metrics(metrics: pd.DataFrame, path: Path) -> Path:
    agg_spec = {
        "mae_mean": ("mae", "mean"),
        "mae_std": ("mae", "std"),
        "rmse_mean": ("rmse", "mean"),
        "rmse_std": ("rmse", "std"),
        "mape_mean": ("mape", "mean"),
        "mape_std": ("mape", "std"),
        "picp_mean": ("picp", "mean"),
        "picp_std": ("picp", "std"),
        "mpiw_mean": ("mpiw", "mean"),
        "mpiw_std": ("mpiw", "std"),
    }
    if "ece" in metrics.columns:
        agg_spec["ece_mean"] = ("ece", "mean")
        agg_spec["ece_std"] = ("ece", "std")
    if "nll" in metrics.columns:
        agg_spec["nll_mean"] = ("nll", "mean")
        agg_spec["nll_std"] = ("nll", "std")
    if "physics_residual_abs" in metrics.columns:
        agg_spec["physics_residual_abs_mean"] = ("physics_residual_abs", "mean")
        agg_spec["physics_residual_abs_std"] = ("physics_residual_abs", "std")
    summary = (
        metrics.groupby([c for c in ["m", "method", "variant"] if c in metrics.columns], as_index=False)
        .agg(**agg_spec)
    )
    summary.to_csv(path, index=False)
    return path


def run_rq2_physics_value_experiment(cfg: ExperimentConfig, epochs: int | None = None, repeats: int = 5) -> Path:
    """RQ2: compare no-physics model against fixed-weight PINN."""
    from .train import train_single_model
    from .uq_methods import predict_deterministic

    out_dir = cfg.out_dir / "rq2_physics_value_nlstt300"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_nlstt_long()
    ids = balanced_sample_trajectories(df, n=cfg.cohort.hmc_n, seed=cfg.cohort.seed + cfg.cohort.hmc_n)
    split = split_trajectory_ids(ids, seed=cfg.cohort.seed + cfg.cohort.hmc_n)
    train_cfg = cfg.train if epochs is None else cfg.train.__class__(**{**cfg.train.__dict__, "epochs": epochs})

    rows = []
    for repeat in range(repeats):
        repeat_seed = 70000 * repeat
        for m in [1, 2]:
            task_dir = out_dir / f"repeat{repeat + 1}" / f"m{m}"
            task_dir.mkdir(parents=True, exist_ok=True)
            train_df = make_prediction_task(df, split.train, m=m)
            val_df = make_prediction_task(df, split.val, m=m)
            test_df = make_prediction_task(df, split.test, m=m)

            methods = {
                "no_physics": train_single_model(train_df, train_cfg, method="no_physics", seed=repeat_seed + 10 + m),
                "fixed_pinn": train_single_model(train_df, train_cfg, method="fixed_pinn", seed=repeat_seed + 20 + m),
            }
            for method, trained in methods.items():
                val_pred = predict_deterministic(trained, val_df)
                test_pred = predict_deterministic(trained, test_df)
                test_pred.to_csv(task_dir / f"pred_{method}_raw.csv", index=False)
                rows.append({"repeat": repeat + 1, "m": m, "method": method, "variant": "raw", **summarize_predictions(test_pred)})
                calibrated, q = residual_calibrate_intervals(val_pred, test_pred, alpha=0.05)
                calibrated["calibration_q"] = q
                calibrated.to_csv(task_dir / f"pred_{method}_calibrated.csv", index=False)
                rows.append(
                    {"repeat": repeat + 1, "m": m, "method": method, "variant": "calibrated", **summarize_predictions(calibrated)}
                )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    return _summarize_repeated_metrics(metrics, out_dir / "metrics_summary.csv")


def run_calibration_experiment(cfg: ExperimentConfig, epochs: int | None = None, repeats: int = 5) -> Path:
    """Compare raw and validation-calibrated intervals on the fair NLSTt-300 m=1/m=2 tasks."""
    from .train import train_ensemble, train_single_model
    from .uq_methods import predict_deterministic, predict_ensemble, predict_mc_dropout

    out_dir = cfg.out_dir / "calibration_nlstt300"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_nlstt_long()
    ids = balanced_sample_trajectories(df, n=cfg.cohort.hmc_n, seed=cfg.cohort.seed + cfg.cohort.hmc_n)
    split = split_trajectory_ids(ids, seed=cfg.cohort.seed + cfg.cohort.hmc_n)
    train_cfg = cfg.train if epochs is None else cfg.train.__class__(**{**cfg.train.__dict__, "epochs": epochs})

    rows = []
    for repeat in range(repeats):
        repeat_seed = 20000 * repeat
        for m in [1, 2]:
            task_dir = out_dir / f"repeat{repeat + 1}" / f"m{m}"
            task_dir.mkdir(parents=True, exist_ok=True)
            train_df = make_prediction_task(df, split.train, m=m)
            val_df = make_prediction_task(df, split.val, m=m)
            test_df = make_prediction_task(df, split.test, m=m)

            fixed = train_single_model(train_df, train_cfg, method="fixed_pinn", seed=repeat_seed + 10 + m)
            adaptive = train_single_model(train_df, train_cfg, method="adaptive_pinn", seed=repeat_seed + 20 + m)
            dropout = train_single_model(train_df, train_cfg, method="mc_dropout", seed=repeat_seed + 30 + m)
            ensemble = train_ensemble(train_df, train_cfg, seed=repeat_seed + 40 + m)

            raw_pairs = {
                "fixed_pinn": (
                    predict_deterministic(fixed, val_df),
                    predict_deterministic(fixed, test_df),
                    "residual",
                ),
                "adaptive_pinn": (
                    predict_mc_dropout(adaptive, val_df, samples=train_cfg.mc_samples),
                    predict_mc_dropout(adaptive, test_df, samples=train_cfg.mc_samples),
                    "interval",
                ),
                "mc_dropout_pinn": (
                    predict_mc_dropout(dropout, val_df, samples=train_cfg.mc_samples),
                    predict_mc_dropout(dropout, test_df, samples=train_cfg.mc_samples),
                    "interval",
                ),
                "deep_ensemble_pinn": (
                    predict_ensemble(ensemble, val_df),
                    predict_ensemble(ensemble, test_df),
                    "interval",
                ),
            }
            for method, (val_pred, test_pred, cal_type) in raw_pairs.items():
                test_pred.to_csv(task_dir / f"pred_{method}_raw.csv", index=False)
                rows.append({"repeat": repeat + 1, "m": m, "method": method, "variant": "raw", **summarize_predictions(test_pred)})
                if cal_type == "interval":
                    calibrated, q = conformalize_existing_intervals(val_pred, test_pred, alpha=0.05)
                else:
                    calibrated, q = residual_calibrate_intervals(val_pred, test_pred, alpha=0.05)
                calibrated["calibration_q"] = q
                calibrated.to_csv(task_dir / f"pred_{method}_calibrated.csv", index=False)
                rows.append({"repeat": repeat + 1, "m": m, "method": method, "variant": "calibrated", **summarize_predictions(calibrated)})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    return _summarize_repeated_metrics(metrics, out_dir / "metrics_summary.csv")


def run_rq1_uq_comparison(cfg: ExperimentConfig, epochs: int | None = None, repeats: int = 5) -> Path:
    """RQ1: compare UQ methods on fair NLSTt-300 m=1/m=2 prediction tasks."""
    from .train import train_ensemble, train_single_model
    from .hmc_baseline import predict_hmc_gompertz
    from .uq_methods import (
        predict_deterministic,
        predict_ensemble,
        predict_laplace_approx,
        predict_mc_dropout,
    )

    out_dir = cfg.out_dir / "rq1_uq_comparison_nlstt300"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_nlstt_long()
    ids = balanced_sample_trajectories(df, n=cfg.cohort.hmc_n, seed=cfg.cohort.seed + cfg.cohort.hmc_n)
    split = split_trajectory_ids(ids, seed=cfg.cohort.seed + cfg.cohort.hmc_n)
    train_cfg = cfg.train if epochs is None else cfg.train.__class__(**{**cfg.train.__dict__, "epochs": epochs})

    rows = []
    for repeat in range(repeats):
        repeat_seed = 60000 * repeat
        for m in [1, 2]:
            task_dir = out_dir / f"repeat{repeat + 1}" / f"m{m}"
            task_dir.mkdir(parents=True, exist_ok=True)
            train_df = make_prediction_task(df, split.train, m=m)
            val_df = make_prediction_task(df, split.val, m=m)
            test_df = make_prediction_task(df, split.test, m=m)

            deterministic = train_single_model(train_df, train_cfg, method="no_physics", seed=repeat_seed + 10 + m)
            dropout = train_single_model(train_df, train_cfg, method="mc_dropout", seed=repeat_seed + 20 + m)
            bayes_map = train_single_model(train_df, train_cfg, method="fixed_pinn", seed=repeat_seed + 30 + m)
            ensemble = train_ensemble(train_df, train_cfg, seed=repeat_seed + 40 + m)

            raw_pairs = {
                "deterministic": (
                    predict_deterministic(deterministic, val_df),
                    predict_deterministic(deterministic, test_df),
                    "residual",
                ),
                "mc_dropout": (
                    predict_mc_dropout(dropout, val_df, samples=train_cfg.mc_samples),
                    predict_mc_dropout(dropout, test_df, samples=train_cfg.mc_samples),
                    "interval",
                ),
                "deep_ensemble": (
                    predict_ensemble(ensemble, val_df),
                    predict_ensemble(ensemble, test_df),
                    "interval",
                ),
                "bayesian_laplace": (
                    predict_laplace_approx(bayes_map, train_df, val_df),
                    predict_laplace_approx(bayes_map, train_df, test_df),
                    "interval",
                ),
                "bayesian_hmc": (
                    predict_hmc_gompertz(train_df, val_df, seed=repeat_seed + 50 + m, num_samples=180, burn_in=80),
                    predict_hmc_gompertz(train_df, test_df, seed=repeat_seed + 60 + m, num_samples=180, burn_in=80),
                    "interval",
                ),
            }

            for method, (val_pred, test_pred, cal_type) in raw_pairs.items():
                test_pred.to_csv(task_dir / f"pred_{method}_raw.csv", index=False)
                rows.append({"repeat": repeat + 1, "m": m, "method": method, "variant": "raw", **summarize_predictions(test_pred)})
                if cal_type == "interval":
                    calibrated, q = conformalize_existing_intervals(val_pred, test_pred, alpha=0.05)
                else:
                    calibrated, q = residual_calibrate_intervals(val_pred, test_pred, alpha=0.05)
                calibrated["calibration_q"] = q
                calibrated.to_csv(task_dir / f"pred_{method}_calibrated.csv", index=False)
                rows.append(
                    {"repeat": repeat + 1, "m": m, "method": method, "variant": "calibrated", **summarize_predictions(calibrated)}
                )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    return _summarize_repeated_metrics(metrics, out_dir / "metrics_summary.csv")


def run_lambda_sweep_experiment(cfg: ExperimentConfig, epochs: int | None = None, repeats: int = 5) -> Path:
    """Run fixed-lambda and adaptive-weight ablations on the main m=2 prediction task."""
    from .train import train_single_model
    from .uq_methods import predict_deterministic, predict_mc_dropout

    out_dir = cfg.out_dir / "lambda_sweep_nlstt300_m2"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_nlstt_long()
    ids = balanced_sample_trajectories(df, n=cfg.cohort.hmc_n, seed=cfg.cohort.seed + cfg.cohort.hmc_n)
    split = split_trajectory_ids(ids, seed=cfg.cohort.seed + cfg.cohort.hmc_n)
    base_cfg = cfg.train if epochs is None else cfg.train.__class__(**{**cfg.train.__dict__, "epochs": epochs})
    fixed_lambdas = [0.0, 0.1, 1.0, 10.0, 100.0]
    adaptive_betas = [1.0, 5.0, 10.0]
    rows = []

    for repeat in range(repeats):
        repeat_seed = 30000 * repeat
        train_df = make_prediction_task(df, split.train, m=2)
        test_df = make_prediction_task(df, split.test, m=2)
        for lam in fixed_lambdas:
            method = "no_physics" if lam == 0.0 else "fixed_pinn"
            train_cfg = base_cfg.__class__(**{**base_cfg.__dict__, "fixed_lambda_phys": lam})
            trained = train_single_model(train_df, train_cfg, method=method, seed=repeat_seed + int(lam * 10) + 1)
            pred = predict_deterministic(trained, test_df)
            pred.to_csv(out_dir / f"repeat{repeat + 1}_fixed_lambda_{lam:g}.csv", index=False)
            rows.append(
                {
                    "repeat": repeat + 1,
                    "m": 2,
                    "method": "fixed_lambda",
                    "variant": f"lambda={lam:g}",
                    **summarize_predictions(pred),
                }
            )
        for beta in adaptive_betas:
            train_cfg = base_cfg.__class__(**{**base_cfg.__dict__, "adaptive_beta": beta})
            trained = train_single_model(train_df, train_cfg, method="adaptive_pinn", seed=repeat_seed + int(beta * 100) + 2)
            pred = predict_mc_dropout(trained, test_df, samples=train_cfg.mc_samples)
            pred.to_csv(out_dir / f"repeat{repeat + 1}_adaptive_beta_{beta:g}.csv", index=False)
            rows.append(
                {
                    "repeat": repeat + 1,
                    "m": 2,
                    "method": "adaptive",
                    "variant": f"beta={beta:g}",
                    **summarize_predictions(pred),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    return _summarize_repeated_metrics(metrics, out_dir / "metrics_summary.csv")


def run_rq3_adaptive_mechanism_experiment(cfg: ExperimentConfig, epochs: int | None = None, repeats: int = 5) -> Path:
    """RQ3: validate uncertainty-adaptive physics weighting against fixed and control weights."""
    from .train import train_single_model
    from .uq_methods import predict_deterministic, predict_mc_dropout

    out_dir = cfg.out_dir / "rq3_adaptive_mechanism_nlstt300"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_nlstt_long()
    ids = balanced_sample_trajectories(df, n=cfg.cohort.hmc_n, seed=cfg.cohort.seed + cfg.cohort.hmc_n)
    split = split_trajectory_ids(ids, seed=cfg.cohort.seed + cfg.cohort.hmc_n)
    base_cfg = cfg.train if epochs is None else cfg.train.__class__(**{**cfg.train.__dict__, "epochs": epochs})

    # Covers 3A, 3B, 3C, and 3D.
    specs: list[dict] = []
    for lam in [0.0, 0.1, 1.0, 10.0, 100.0]:
        specs.append({"method": "no_physics" if lam == 0.0 else "fixed_pinn", "label": f"fixed_lambda={lam:g}", "cfg": {"fixed_lambda_phys": lam}})
    for beta in [1.0, 5.0, 10.0]:
        specs.append({"method": "adaptive_pinn", "label": f"adaptive_beta={beta:g}", "cfg": {"adaptive_beta": beta}})
    specs.extend(
        [
            {"method": "reliability_gated_pinn", "label": "rg_pinn_uq_full", "cfg": {}},
            {"method": "reliability_uq_only_pinn", "label": "rg_uq_only", "cfg": {}},
            {"method": "reliability_sparsity_only_pinn", "label": "rg_sparsity_only", "cfg": {}},
            {"method": "reliability_phys_only_pinn", "label": "rg_phys_only", "cfg": {}},
        ]
    )
    specs.extend(
        [
            {"method": "epoch_pinn", "label": "epoch_schedule", "cfg": {}},
            {"method": "random_pinn", "label": "random_weight", "cfg": {}},
            {"method": "inverse_adaptive_pinn", "label": "inverse_uncertainty", "cfg": {"adaptive_beta": 5.0}},
        ]
    )

    rows = []
    lambda_rows = []
    for repeat in range(repeats):
        repeat_seed = 80000 * repeat
        for m in [1, 2]:
            train_df = make_prediction_task(df, split.train, m=m)
            test_df = make_prediction_task(df, split.test, m=m)
            for spec_idx, spec in enumerate(specs):
                train_cfg = base_cfg.__class__(**{**base_cfg.__dict__, **spec["cfg"]})
                trained = train_single_model(
                    train_df,
                    train_cfg,
                    method=spec["method"],
                    seed=repeat_seed + spec_idx * 101 + m,
                )
                if spec["method"] in {
                    "adaptive_pinn",
                    "inverse_adaptive_pinn",
                    "reliability_gated_pinn",
                    "reliability_uq_only_pinn",
                    "reliability_sparsity_only_pinn",
                    "reliability_phys_only_pinn",
                }:
                    pred = predict_mc_dropout(trained, test_df, samples=train_cfg.mc_samples)
                else:
                    pred = predict_deterministic(trained, test_df)
                task_dir = out_dir / f"repeat{repeat + 1}" / f"m{m}"
                task_dir.mkdir(parents=True, exist_ok=True)
                pred.to_csv(task_dir / f"pred_{spec['label'].replace('=', '_')}.csv", index=False)
                metric_row = {
                    "repeat": repeat + 1,
                    "m": m,
                    "method": spec["method"],
                    "variant": spec["label"],
                    **summarize_predictions(pred),
                }
                rows.append(metric_row)
                if trained.lambda_log is not None:
                    log = trained.lambda_log.copy()
                    log["repeat"] = repeat + 1
                    log["m"] = m
                    log["method"] = spec["method"]
                    log["variant"] = spec["label"]
                    lambda_rows.append(log)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = _summarize_repeated_metrics(metrics, out_dir / "metrics_summary.csv")
    if lambda_rows:
        lambda_log = pd.concat(lambda_rows, ignore_index=True)
        lambda_log.to_csv(out_dir / "lambda_logs.csv", index=False)
        corr_rows = []
        for keys, group in lambda_log.groupby(["m", "method", "variant"]):
            if group["lambda_phys"].std() > 0 and group["uncertainty_proxy"].std() > 0:
                corr = group["lambda_phys"].corr(group["uncertainty_proxy"])
            else:
                corr = float("nan")
            corr_rows.append(
                {
                    "m": keys[0],
                    "method": keys[1],
                    "variant": keys[2],
                    "lambda_uncertainty_corr": corr,
                    "lambda_mean": group["lambda_phys"].mean(),
                    "uncertainty_proxy_mean": group["uncertainty_proxy"].mean(),
                }
            )
        pd.DataFrame(corr_rows).to_csv(out_dir / "lambda_uncertainty_correlation.csv", index=False)
    return summary


def _write_deeplesion_report(
    out_dir: Path,
    cohort: pd.DataFrame,
    split,
    summary_paths: dict[str, Path],
    target_transform: str = "raw_log_volume",
) -> Path:
    report_path = out_dir / "deeplesion_len5_experiment_report.md"
    n_traj = cohort["trajectory_id"].nunique()
    n_patients = cohort["patient_id"].astype(str).nunique()
    class_counts = (
        cohort.groupby("trajectory_id", as_index=False).agg(growth_class=("growth_class", "first"))["growth_class"]
        .value_counts()
        .to_dict()
    )
    lines = [
        "# DeepLesion/DLT Five-Follow-up Main Experiment",
        "",
        "## Dataset",
        "",
        f"- Trajectories: {n_traj}",
        f"- Patients: {n_patients}",
        "- Follow-ups per trajectory: first 5 real DLT-matched visits",
        "- Target: visit 5 (T4)",
        "- Inputs: m=1,2,3,4 observed visits",
        f"- Target transform: {target_transform}",
        "- Size variable: RECIST ellipsoid volume proxy",
        f"- Split: patient-level train={len(split.train)}, val={len(split.val)}, test={len(split.test)} trajectories",
        f"- Growth class counts: {json.dumps(class_counts, ensure_ascii=False)}",
        "",
        "## Outputs",
        "",
    ]
    for name, path in summary_paths.items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "DeepLesion/DLT is a multi-organ CT lesion dataset, not a lung-nodule-only dataset. "
            "Use it as the main real multi-follow-up lesion cohort, and keep NLSTt as an external lung-nodule validation cohort.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _markdown_table(frame: pd.DataFrame, cols: list[str] | None = None) -> str:
    view = frame if cols is None else frame[cols]
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    return "\n".join(lines)


def run_deeplesion_len5_main_experiment(
    cfg: ExperimentConfig,
    epochs: int | None = None,
    repeats: int = 3,
    relative_log: bool = False,
    out_name: str | None = None,
) -> Path:
    """Run RQ1/RQ2/RQ3 on 205 real five-follow-up DeepLesion/DLT trajectories."""
    from .train import train_ensemble, train_single_model
    from .uq_methods import (
        predict_deterministic,
        predict_ensemble,
        predict_laplace_approx,
        predict_mc_dropout,
    )

    if out_name is None:
        out_name = "deeplesion_len5_relative_main205" if relative_log else "deeplesion_len5_main205"
    out_dir = cfg.out_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_deeplesion_len5_long(relative_log=relative_log)
    ids = sorted(df["trajectory_id"].astype(str).unique().tolist())
    split = split_trajectory_ids_by_patient(
        df,
        ids,
        seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )
    df.to_csv(out_dir / "cohort_deeplesion_len5_long.csv", index=False)
    pd.DataFrame({"trajectory_id": split.train, "split": "train"}).to_csv(out_dir / "split_train.csv", index=False)
    pd.DataFrame({"trajectory_id": split.val, "split": "val"}).to_csv(out_dir / "split_val.csv", index=False)
    pd.DataFrame({"trajectory_id": split.test, "split": "test"}).to_csv(out_dir / "split_test.csv", index=False)

    train_cfg = cfg.train if epochs is None else cfg.train.__class__(**{**cfg.train.__dict__, "epochs": epochs})
    summary_paths: dict[str, Path] = {}

    rq1_dir = out_dir / "rq1_uq_comparison"
    rq1_dir.mkdir(exist_ok=True)
    rq1_rows = []
    for repeat in range(repeats):
        repeat_seed = 61000 * repeat
        for m in [1, 2, 3, 4]:
            task_dir = rq1_dir / f"repeat{repeat + 1}" / f"m{m}"
            task_dir.mkdir(parents=True, exist_ok=True)
            train_df = make_deeplesion_prediction_task(df, split.train, m=m)
            val_df = make_deeplesion_prediction_task(df, split.val, m=m)
            test_df = make_deeplesion_prediction_task(df, split.test, m=m)
            pd.concat(
                [train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")],
                ignore_index=True,
            ).to_csv(task_dir / "task_rows.csv", index=False)

            deterministic = train_single_model(train_df, train_cfg, method="no_physics", seed=repeat_seed + 10 + m)
            dropout = train_single_model(train_df, train_cfg, method="mc_dropout", seed=repeat_seed + 20 + m)
            bayes_map = train_single_model(train_df, train_cfg, method="fixed_pinn", seed=repeat_seed + 30 + m)
            ensemble = train_ensemble(train_df, train_cfg, seed=repeat_seed + 40 + m)

            raw_pairs = {
                "deterministic": (
                    predict_deterministic(deterministic, val_df),
                    predict_deterministic(deterministic, test_df),
                    "residual",
                ),
                "mc_dropout": (
                    predict_mc_dropout(dropout, val_df, samples=train_cfg.mc_samples),
                    predict_mc_dropout(dropout, test_df, samples=train_cfg.mc_samples),
                    "interval",
                ),
                "deep_ensemble": (
                    predict_ensemble(ensemble, val_df),
                    predict_ensemble(ensemble, test_df),
                    "interval",
                ),
                "bayesian_laplace": (
                    predict_laplace_approx(bayes_map, train_df, val_df),
                    predict_laplace_approx(bayes_map, train_df, test_df),
                    "interval",
                ),
            }
            for method, (val_pred, test_pred, cal_type) in raw_pairs.items():
                test_pred.to_csv(task_dir / f"pred_{method}_raw.csv", index=False)
                rq1_rows.append({"repeat": repeat + 1, "m": m, "method": method, "variant": "raw", **summarize_predictions(test_pred)})
                if cal_type == "interval":
                    calibrated, q = conformalize_existing_intervals(val_pred, test_pred, alpha=0.05)
                else:
                    calibrated, q = residual_calibrate_intervals(val_pred, test_pred, alpha=0.05)
                calibrated["calibration_q"] = q
                calibrated.to_csv(task_dir / f"pred_{method}_calibrated.csv", index=False)
                rq1_rows.append({"repeat": repeat + 1, "m": m, "method": method, "variant": "calibrated", **summarize_predictions(calibrated)})

    rq1_metrics = pd.DataFrame(rq1_rows)
    rq1_metrics.to_csv(rq1_dir / "metrics.csv", index=False)
    summary_paths["RQ1 UQ comparison"] = _summarize_repeated_metrics(rq1_metrics, rq1_dir / "metrics_summary.csv")

    rq2_dir = out_dir / "rq2_physics_value"
    rq2_dir.mkdir(exist_ok=True)
    rq2_rows = []
    for repeat in range(repeats):
        repeat_seed = 72000 * repeat
        for m in [1, 2, 3, 4]:
            task_dir = rq2_dir / f"repeat{repeat + 1}" / f"m{m}"
            task_dir.mkdir(parents=True, exist_ok=True)
            train_df = make_deeplesion_prediction_task(df, split.train, m=m)
            val_df = make_deeplesion_prediction_task(df, split.val, m=m)
            test_df = make_deeplesion_prediction_task(df, split.test, m=m)
            methods = {
                "no_physics": train_single_model(train_df, train_cfg, method="no_physics", seed=repeat_seed + 10 + m),
                "fixed_pinn": train_single_model(train_df, train_cfg, method="fixed_pinn", seed=repeat_seed + 20 + m),
            }
            for method, trained in methods.items():
                val_pred = predict_deterministic(trained, val_df)
                test_pred = predict_deterministic(trained, test_df)
                test_pred.to_csv(task_dir / f"pred_{method}_raw.csv", index=False)
                rq2_rows.append({"repeat": repeat + 1, "m": m, "method": method, "variant": "raw", **summarize_predictions(test_pred)})
                calibrated, q = residual_calibrate_intervals(val_pred, test_pred, alpha=0.05)
                calibrated["calibration_q"] = q
                calibrated.to_csv(task_dir / f"pred_{method}_calibrated.csv", index=False)
                rq2_rows.append({"repeat": repeat + 1, "m": m, "method": method, "variant": "calibrated", **summarize_predictions(calibrated)})
    rq2_metrics = pd.DataFrame(rq2_rows)
    rq2_metrics.to_csv(rq2_dir / "metrics.csv", index=False)
    summary_paths["RQ2 physics value"] = _summarize_repeated_metrics(rq2_metrics, rq2_dir / "metrics_summary.csv")

    rq3_dir = out_dir / "rq3_adaptive_mechanism"
    rq3_dir.mkdir(exist_ok=True)
    specs: list[dict] = []
    for lam in [0.0, 0.1, 1.0, 10.0, 100.0]:
        specs.append({"method": "no_physics" if lam == 0.0 else "fixed_pinn", "label": f"fixed_lambda={lam:g}", "cfg": {"fixed_lambda_phys": lam}})
    for beta in [1.0, 5.0, 10.0]:
        specs.append({"method": "adaptive_pinn", "label": f"adaptive_beta={beta:g}", "cfg": {"adaptive_beta": beta}})
    specs.extend(
        [
            {"method": "epoch_pinn", "label": "epoch_schedule", "cfg": {}},
            {"method": "random_pinn", "label": "random_weight", "cfg": {}},
            {"method": "inverse_adaptive_pinn", "label": "inverse_uncertainty", "cfg": {"adaptive_beta": 5.0}},
        ]
    )
    rq3_rows = []
    lambda_rows = []
    for repeat in range(repeats):
        repeat_seed = 83000 * repeat
        for m in [1, 2, 3, 4]:
            train_df = make_deeplesion_prediction_task(df, split.train, m=m)
            test_df = make_deeplesion_prediction_task(df, split.test, m=m)
            task_dir = rq3_dir / f"repeat{repeat + 1}" / f"m{m}"
            task_dir.mkdir(parents=True, exist_ok=True)
            for spec_idx, spec in enumerate(specs):
                spec_cfg = train_cfg.__class__(**{**train_cfg.__dict__, **spec["cfg"]})
                trained = train_single_model(train_df, spec_cfg, method=spec["method"], seed=repeat_seed + spec_idx * 101 + m)
                if spec["method"] in {"adaptive_pinn", "inverse_adaptive_pinn"}:
                    pred = predict_mc_dropout(trained, test_df, samples=spec_cfg.mc_samples)
                else:
                    pred = predict_deterministic(trained, test_df)
                pred.to_csv(task_dir / f"pred_{spec['label'].replace('=', '_')}.csv", index=False)
                rq3_rows.append({"repeat": repeat + 1, "m": m, "method": spec["method"], "variant": spec["label"], **summarize_predictions(pred)})
                if trained.lambda_log is not None:
                    log = trained.lambda_log.copy()
                    log["repeat"] = repeat + 1
                    log["m"] = m
                    log["method"] = spec["method"]
                    log["variant"] = spec["label"]
                    lambda_rows.append(log)
    rq3_metrics = pd.DataFrame(rq3_rows)
    rq3_metrics.to_csv(rq3_dir / "metrics.csv", index=False)
    if lambda_rows:
        pd.concat(lambda_rows, ignore_index=True).to_csv(rq3_dir / "lambda_logs.csv", index=False)
    summary_paths["RQ3 adaptive mechanism"] = _summarize_repeated_metrics(rq3_metrics, rq3_dir / "metrics_summary.csv")

    return _write_deeplesion_report(
        out_dir,
        df,
        split,
        summary_paths,
        target_transform="relative log(V/V0)" if relative_log else "raw log(V)",
    )


def run_deeplesion_subgroup_gompertz_experiment(cfg: ExperimentConfig) -> Path:
    """Run subgroup staged mechanistic robustness tables on raw DeepLesion log-volume."""
    from .mechanistic_baselines import predict_density_staged_mechanistic, population_growth_prior

    out_dir = cfg.out_dir / "deeplesion_len5_subgroup_staged_mechanistic"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_deeplesion_len5_long(relative_log=False)
    df = df.copy()
    df["generator"] = "real_deeplesion_dlt"
    df["total_nodes"] = 5
    df["subgroup"] = df["body_region_group"].where(
        df["body_region_group"].isin(["chest_or_lung", "abdomen_or_liver"]),
        "other",
    )
    ids = sorted(df["trajectory_id"].astype(str).unique().tolist())
    split = split_trajectory_ids_by_patient(
        df,
        ids,
        seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )

    rows = []
    pred_frames = []
    subgroup_counts = (
        df.groupby("trajectory_id", as_index=False).agg(subgroup=("subgroup", "first"))["subgroup"]
        .value_counts()
        .reset_index()
    )
    subgroup_counts.columns = ["subgroup", "n_trajectories"]
    subgroup_counts.to_csv(out_dir / "subgroup_counts.csv", index=False)

    for subgroup in ["chest_or_lung", "abdomen_or_liver", "other"]:
        subgroup_ids = sorted(df.loc[df["subgroup"] == subgroup, "trajectory_id"].astype(str).unique().tolist())
        train_ids = sorted(set(split.train).intersection(subgroup_ids))
        test_ids = sorted(set(split.test).intersection(subgroup_ids))
        if not train_ids or not test_ids:
            continue
        prior = population_growth_prior(df[df["subgroup"] == subgroup], train_ids)
        for m in [1, 2, 3, 4]:
            pred = predict_density_staged_mechanistic(
                df[df["subgroup"] == subgroup],
                test_ids,
                k=m,
                prior_slope=prior,
            )
            pred["subgroup"] = subgroup
            pred["m"] = m
            pred["train_n"] = len(train_ids)
            pred["test_n"] = len(test_ids)
            pred["prior_slope"] = prior
            model_name = str(pred["fit_model"].iloc[0]) if not pred.empty else "unknown"
            pred.to_csv(out_dir / f"pred_{subgroup}_m{m}_{model_name}.csv", index=False)
            pred_frames.append(pred)
            rows.append(
                {
                    "subgroup": subgroup,
                    "m": m,
                    "model": model_name,
                    "train_n": len(train_ids),
                    "test_n": len(test_ids),
                    "prior_slope": prior,
                    **summarize_predictions(pred),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "subgroup_staged_mechanistic_metrics.csv", index=False)
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / "subgroup_staged_mechanistic_predictions.csv", index=False)

    pivot = metrics.pivot(index="subgroup", columns="m", values="rmse").reset_index()
    pivot.columns = ["subgroup"] + [f"m{int(c)}_rmse" for c in pivot.columns[1:]]
    pivot.to_csv(out_dir / "subgroup_staged_mechanistic_rmse_table.csv", index=False)

    lines = [
        "# DeepLesion Subgroup Staged Mechanistic Robustness",
        "",
        "## Setting",
        "",
        "- Raw log-volume target, no relative normalization.",
        "- m=1 uses persistence because individual growth rate is not identifiable from one point.",
        "- m=2 uses log-linear extrapolation because two points identify a slope.",
        "- m=3,4 use Gompertz because nonlinear curvature becomes partially identifiable.",
        "",
        "## Subgroup Counts",
        "",
        _markdown_table(subgroup_counts),
        "",
        "## RMSE Table",
        "",
        _markdown_table(pivot),
        "",
        "## Full Metrics",
        "",
        _markdown_table(metrics[["subgroup", "m", "model", "train_n", "test_n", "mae", "rmse", "mape"]]),
        "",
    ]
    report = out_dir / "subgroup_staged_mechanistic_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_deeplesion_reliability_gated_experiment(
    cfg: ExperimentConfig,
    epochs: int | None = None,
    repeats: int = 3,
) -> Path:
    """Focused RQ3 experiment for reliability-gated physics weighting on relative DeepLesion."""
    from .train import train_single_model
    from .uq_methods import predict_deterministic, predict_mc_dropout

    out_dir = cfg.out_dir / "deeplesion_len5_relative_rg_pinn"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_deeplesion_len5_long(relative_log=True)
    ids = sorted(df["trajectory_id"].astype(str).unique().tolist())
    split = split_trajectory_ids_by_patient(
        df,
        ids,
        seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )
    train_cfg = cfg.train if epochs is None else cfg.train.__class__(**{**cfg.train.__dict__, "epochs": epochs})

    specs = [
        {"method": "no_physics", "label": "fixed_lambda=0", "cfg": {"fixed_lambda_phys": 0.0}},
        {"method": "fixed_pinn", "label": "fixed_lambda=1", "cfg": {"fixed_lambda_phys": 1.0}},
        {"method": "fixed_pinn", "label": "fixed_lambda=10", "cfg": {"fixed_lambda_phys": 10.0}},
        {"method": "fixed_pinn", "label": "fixed_lambda=100", "cfg": {"fixed_lambda_phys": 100.0}},
        {"method": "adaptive_pinn", "label": "uncertainty_adaptive_beta=5", "cfg": {"adaptive_beta": 5.0}},
        {"method": "epoch_pinn", "label": "epoch_schedule", "cfg": {}},
        {"method": "random_pinn", "label": "random_weight", "cfg": {}},
        {"method": "inverse_adaptive_pinn", "label": "inverse_uncertainty", "cfg": {"adaptive_beta": 5.0}},
        {"method": "reliability_uq_only_pinn", "label": "rg_uq_only", "cfg": {}},
        {"method": "reliability_sparsity_only_pinn", "label": "rg_sparsity_only", "cfg": {}},
        {"method": "reliability_phys_only_pinn", "label": "rg_phys_only", "cfg": {}},
        {"method": "reliability_gated_pinn", "label": "rg_pinn_uq_full", "cfg": {}},
        {
            "method": "reliability_boosted_pinn",
            "label": "rg_boosted_uq",
            "cfg": {"adaptive_lambda_max": 50.0, "reliability_uq_beta": 2.0},
        },
    ]
    stochastic = {
        "adaptive_pinn",
        "inverse_adaptive_pinn",
        "reliability_uq_only_pinn",
        "reliability_sparsity_only_pinn",
        "reliability_phys_only_pinn",
        "reliability_gated_pinn",
        "reliability_boosted_pinn",
    }
    rows = []
    lambda_rows = []
    for repeat in range(repeats):
        repeat_seed = 93000 * repeat
        for m in [1, 2, 3, 4]:
            train_df = make_deeplesion_prediction_task(df, split.train, m=m)
            test_df = make_deeplesion_prediction_task(df, split.test, m=m)
            task_dir = out_dir / f"repeat{repeat + 1}" / f"m{m}"
            task_dir.mkdir(parents=True, exist_ok=True)
            for spec_idx, spec in enumerate(specs):
                spec_cfg = train_cfg.__class__(**{**train_cfg.__dict__, **spec["cfg"]})
                trained = train_single_model(
                    train_df,
                    spec_cfg,
                    method=spec["method"],
                    seed=repeat_seed + spec_idx * 101 + m,
                )
                if spec["method"] in stochastic:
                    pred = predict_mc_dropout(trained, test_df, samples=spec_cfg.mc_samples)
                else:
                    pred = predict_deterministic(trained, test_df)
                pred.to_csv(task_dir / f"pred_{spec['label'].replace('=', '_')}.csv", index=False)
                rows.append(
                    {
                        "repeat": repeat + 1,
                        "m": m,
                        "method": spec["method"],
                        "variant": spec["label"],
                        **summarize_predictions(pred),
                    }
                )
                if trained.lambda_log is not None:
                    log = trained.lambda_log.copy()
                    log["repeat"] = repeat + 1
                    log["m"] = m
                    log["method"] = spec["method"]
                    log["variant"] = spec["label"]
                    lambda_rows.append(log)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = _summarize_repeated_metrics(metrics, out_dir / "metrics_summary.csv")
    if lambda_rows:
        pd.concat(lambda_rows, ignore_index=True).to_csv(out_dir / "lambda_logs.csv", index=False)

    best = metrics.groupby(["m", "variant"], as_index=False).agg(rmse=("rmse", "mean"))
    best = best.sort_values(["m", "rmse"]).groupby("m", as_index=False).head(5)
    lines = [
        "# DeepLesion Relative RG-PINN-UQ Focused Experiment",
        "",
        f"- Target: relative log(V/V0)",
        f"- Repeats: {repeats}",
        f"- Epochs: {train_cfg.epochs}",
        "",
        "## Top RMSE Variants By m",
        "",
        _markdown_table(best),
        "",
        f"Full summary: `{summary}`",
        "",
    ]
    report = out_dir / "rg_pinn_uq_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_deeplesion_mixture_physics_experiment(
    cfg: ExperimentConfig,
    epochs: int | None = None,
    repeats: int = 3,
) -> Path:
    """Focused RQ3 replacement: individualized mixture physics on relative DeepLesion."""
    from .train import train_single_model
    from .uq_methods import predict_deterministic, predict_mc_dropout

    out_dir = cfg.out_dir / "deeplesion_len5_relative_mixture_physics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_deeplesion_len5_long(relative_log=True)
    ids = sorted(df["trajectory_id"].astype(str).unique().tolist())
    split = split_trajectory_ids_by_patient(
        df,
        ids,
        seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )
    df.to_csv(out_dir / "cohort_deeplesion_len5_relative_long.csv", index=False)
    train_cfg = cfg.train if epochs is None else cfg.train.__class__(**{**cfg.train.__dict__, "epochs": epochs})
    specs = [
        {"method": "no_physics", "label": "no_physics", "cfg": {"fixed_lambda_phys": 0.0}, "stochastic": False},
        {"method": "fixed_pinn", "label": "fixed_gompertz_lambda=10", "cfg": {"fixed_lambda_phys": 10.0}, "stochastic": False},
        {"method": "adaptive_pinn", "label": "uncertainty_adaptive_beta=5", "cfg": {"adaptive_beta": 5.0}, "stochastic": True},
        {"method": "mixture_physics_pinn", "label": "mixture_physics", "cfg": {"fixed_lambda_phys": 10.0}, "stochastic": False},
        {"method": "mixture_physics_uq", "label": "mixture_physics_uq", "cfg": {"fixed_lambda_phys": 10.0}, "stochastic": True},
    ]

    rows = []
    mixture_rows = []
    lambda_rows = []
    for repeat in range(repeats):
        repeat_seed = 104000 * repeat
        for m in [1, 2, 3, 4]:
            train_df = make_deeplesion_prediction_task(df, split.train, m=m)
            test_df = make_deeplesion_prediction_task(df, split.test, m=m)
            task_dir = out_dir / f"repeat{repeat + 1}" / f"m{m}"
            task_dir.mkdir(parents=True, exist_ok=True)
            pd.concat(
                [train_df.assign(split="train"), test_df.assign(split="test")],
                ignore_index=True,
            ).to_csv(task_dir / "task_rows.csv", index=False)
            for spec_idx, spec in enumerate(specs):
                spec_cfg = train_cfg.__class__(**{**train_cfg.__dict__, **spec["cfg"]})
                trained = train_single_model(
                    train_df,
                    spec_cfg,
                    method=spec["method"],
                    seed=repeat_seed + spec_idx * 101 + m,
                )
                if spec["stochastic"]:
                    pred = predict_mc_dropout(trained, test_df, samples=spec_cfg.mc_samples)
                else:
                    pred = predict_deterministic(trained, test_df)
                pred.to_csv(task_dir / f"pred_{spec['label'].replace('=', '_')}.csv", index=False)
                rows.append(
                    {
                        "repeat": repeat + 1,
                        "m": m,
                        "method": spec["method"],
                        "variant": spec["label"],
                        **summarize_predictions(pred),
                    }
                )
                if spec["method"].startswith("mixture_physics") and "mix_selected" in pred.columns:
                    mix_counts = pred["mix_selected"].value_counts(normalize=True).rename_axis("selected_physics").reset_index(name="fraction")
                    for _, row in mix_counts.iterrows():
                        mixture_rows.append(
                            {
                                "repeat": repeat + 1,
                                "m": m,
                                "variant": spec["label"],
                                "selected_physics": row["selected_physics"],
                                "fraction": float(row["fraction"]),
                            }
                        )
                if trained.lambda_log is not None:
                    log = trained.lambda_log.copy()
                    log["repeat"] = repeat + 1
                    log["m"] = m
                    log["method"] = spec["method"]
                    log["variant"] = spec["label"]
                    lambda_rows.append(log)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = _summarize_repeated_metrics(metrics, out_dir / "metrics_summary.csv")
    if mixture_rows:
        mixture = pd.DataFrame(mixture_rows)
        mixture.to_csv(out_dir / "mixture_selection.csv", index=False)
        mixture_summary = (
            mixture.groupby(["m", "variant", "selected_physics"], as_index=False)
            .agg(fraction_mean=("fraction", "mean"), fraction_std=("fraction", "std"))
            .sort_values(["m", "variant", "fraction_mean"], ascending=[True, True, False])
        )
        mixture_summary.to_csv(out_dir / "mixture_selection_summary.csv", index=False)
    else:
        mixture_summary = pd.DataFrame()
    if lambda_rows:
        pd.concat(lambda_rows, ignore_index=True).to_csv(out_dir / "lambda_logs.csv", index=False)

    best = metrics.groupby(["m", "variant"], as_index=False).agg(rmse=("rmse", "mean"))
    best = best.sort_values(["m", "rmse"])
    lines = [
        "# DeepLesion Relative Mixture Physics Experiment",
        "",
        "- Target: relative log(V/V0)",
        "- Candidate physics: Gompertz, Logistic, Exponential, Decay, Stable",
        f"- Repeats: {repeats}",
        f"- Epochs: {train_cfg.epochs}",
        "",
        "## RMSE By Method",
        "",
        _markdown_table(best),
        "",
        "## Mixture Selection Summary",
        "",
        _markdown_table(mixture_summary) if not mixture_summary.empty else "No mixture selection rows.",
        "",
        f"Full metrics: `{summary}`",
        "",
    ]
    report = out_dir / "mixture_physics_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_deeplesion_feature_fusion_experiment(
    cfg: ExperimentConfig,
    epochs: int | None = None,
    repeats: int = 3,
) -> Path:
    """Focused RQ3 candidate: physics-guided feature fusion on relative DeepLesion."""
    from .train import train_single_model
    from .uq_methods import predict_deterministic, predict_mc_dropout

    out_dir = cfg.out_dir / "deeplesion_len5_relative_physics_feature_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_deeplesion_len5_long(relative_log=True)
    ids = sorted(df["trajectory_id"].astype(str).unique().tolist())
    split = split_trajectory_ids_by_patient(
        df,
        ids,
        seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )
    df.to_csv(out_dir / "cohort_deeplesion_len5_relative_long.csv", index=False)
    train_cfg = cfg.train if epochs is None else cfg.train.__class__(**{**cfg.train.__dict__, "epochs": epochs})
    specs = [
        {"method": "no_physics", "label": "plain_mlp", "cfg": {"fixed_lambda_phys": 0.0}, "stochastic": False},
        {"method": "fixed_pinn", "label": "fixed_gompertz_lambda=10", "cfg": {"fixed_lambda_phys": 10.0}, "stochastic": False},
        {"method": "adaptive_pinn", "label": "uncertainty_adaptive_beta=5", "cfg": {"adaptive_beta": 5.0}, "stochastic": True},
        {"method": "physics_feature_fusion", "label": "physics_feature_fusion", "cfg": {"fixed_lambda_phys": 0.0}, "stochastic": False},
        {"method": "physics_feature_fusion_uq", "label": "physics_feature_fusion_uq", "cfg": {"fixed_lambda_phys": 0.0}, "stochastic": True},
        {"method": "physics_feature_fusion", "label": "physics_feature_fusion_pinn_lambda=1", "cfg": {"fixed_lambda_phys": 1.0}, "stochastic": False},
    ]
    rows = []
    lambda_rows = []
    for repeat in range(repeats):
        repeat_seed = 114000 * repeat
        for m in [1, 2, 3, 4]:
            train_df = make_deeplesion_prediction_task(df, split.train, m=m)
            test_df = make_deeplesion_prediction_task(df, split.test, m=m)
            task_dir = out_dir / f"repeat{repeat + 1}" / f"m{m}"
            task_dir.mkdir(parents=True, exist_ok=True)
            pd.concat(
                [train_df.assign(split="train"), test_df.assign(split="test")],
                ignore_index=True,
            ).to_csv(task_dir / "task_rows.csv", index=False)
            for spec_idx, spec in enumerate(specs):
                spec_cfg = train_cfg.__class__(**{**train_cfg.__dict__, **spec["cfg"]})
                trained = train_single_model(
                    train_df,
                    spec_cfg,
                    method=spec["method"],
                    seed=repeat_seed + spec_idx * 101 + m,
                )
                if spec["stochastic"]:
                    pred = predict_mc_dropout(trained, test_df, samples=spec_cfg.mc_samples)
                else:
                    pred = predict_deterministic(trained, test_df)
                pred.to_csv(task_dir / f"pred_{spec['label'].replace('=', '_')}.csv", index=False)
                rows.append(
                    {
                        "repeat": repeat + 1,
                        "m": m,
                        "method": spec["method"],
                        "variant": spec["label"],
                        **summarize_predictions(pred),
                    }
                )
                if trained.lambda_log is not None:
                    log = trained.lambda_log.copy()
                    log["repeat"] = repeat + 1
                    log["m"] = m
                    log["method"] = spec["method"]
                    log["variant"] = spec["label"]
                    lambda_rows.append(log)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = _summarize_repeated_metrics(metrics, out_dir / "metrics_summary.csv")
    if lambda_rows:
        pd.concat(lambda_rows, ignore_index=True).to_csv(out_dir / "lambda_logs.csv", index=False)
    ranked = metrics.groupby(["m", "variant"], as_index=False).agg(rmse=("rmse", "mean"))
    ranked = ranked.sort_values(["m", "rmse"])
    lines = [
        "# DeepLesion Relative Physics Feature Fusion Experiment",
        "",
        "- Target: relative log(V/V0)",
        "- Fusion features: sparse observations plus stable, local-linear, global-linear, shrinkage and slope summaries",
        f"- Repeats: {repeats}",
        f"- Epochs: {train_cfg.epochs}",
        "",
        "## RMSE By Method",
        "",
        _markdown_table(ranked),
        "",
        f"Full metrics: `{summary}`",
        "",
    ]
    report = out_dir / "physics_feature_fusion_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _predict_persistence(task_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in task_df.to_dict("records"):
        pred = float(row["logv_obs"][-1])
        rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "logv_target": float(row["logv_target"]),
                "logv_mean": pred,
                "logv_lower": pred,
                "logv_upper": pred,
                "physics_residual": 0.0,
                "physics_residual_abs": 0.0,
            }
        )
    return pd.DataFrame(rows)


def run_deeplesion_anchored_correction_experiment(
    cfg: ExperimentConfig,
    epochs: int | None = None,
    repeats: int = 3,
) -> Path:
    """Focused RQ3 candidate: sparse CT anchored correction on relative DeepLesion."""
    from .train import train_single_model
    from .uq_methods import predict_deterministic, predict_mc_dropout

    out_dir = cfg.out_dir / "deeplesion_len5_relative_anchored_correction"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_deeplesion_len5_long(relative_log=True)
    ids = sorted(df["trajectory_id"].astype(str).unique().tolist())
    split = split_trajectory_ids_by_patient(
        df,
        ids,
        seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )
    df.to_csv(out_dir / "cohort_deeplesion_len5_relative_long.csv", index=False)
    train_cfg = cfg.train if epochs is None else cfg.train.__class__(**{**cfg.train.__dict__, "epochs": epochs})
    specs = [
        {"method": "no_physics", "label": "plain_mlp", "cfg": {"fixed_lambda_phys": 0.0}, "stochastic": False},
        {"method": "fixed_pinn", "label": "fixed_gompertz_lambda=10", "cfg": {"fixed_lambda_phys": 10.0}, "stochastic": False},
        {"method": "adaptive_pinn", "label": "uncertainty_adaptive_beta=5", "cfg": {"adaptive_beta": 5.0}, "stochastic": True},
        {"method": "anchored_correction", "label": "anchored_correction", "cfg": {"fixed_lambda_phys": 0.0}, "stochastic": False},
        {"method": "anchored_correction_uq", "label": "anchored_correction_uq", "cfg": {"fixed_lambda_phys": 0.0}, "stochastic": True},
        {"method": "anchored_correction_pinn", "label": "anchored_correction_pinn_lambda=1", "cfg": {"fixed_lambda_phys": 1.0}, "stochastic": False},
    ]
    rows = []
    lambda_rows = []
    gate_rows = []
    for repeat in range(repeats):
        repeat_seed = 124000 * repeat
        for m in [1, 2, 3, 4]:
            train_df = make_deeplesion_prediction_task(df, split.train, m=m)
            test_df = make_deeplesion_prediction_task(df, split.test, m=m)
            task_dir = out_dir / f"repeat{repeat + 1}" / f"m{m}"
            task_dir.mkdir(parents=True, exist_ok=True)
            pd.concat(
                [train_df.assign(split="train"), test_df.assign(split="test")],
                ignore_index=True,
            ).to_csv(task_dir / "task_rows.csv", index=False)

            persistence = _predict_persistence(test_df)
            persistence.to_csv(task_dir / "pred_persistence_anchor.csv", index=False)
            rows.append({"repeat": repeat + 1, "m": m, "method": "persistence", "variant": "persistence_anchor", **summarize_predictions(persistence)})

            for spec_idx, spec in enumerate(specs):
                spec_cfg = train_cfg.__class__(**{**train_cfg.__dict__, **spec["cfg"]})
                trained = train_single_model(
                    train_df,
                    spec_cfg,
                    method=spec["method"],
                    seed=repeat_seed + spec_idx * 101 + m,
                )
                if spec["stochastic"]:
                    pred = predict_mc_dropout(trained, test_df, samples=spec_cfg.mc_samples)
                else:
                    pred = predict_deterministic(trained, test_df)
                pred.to_csv(task_dir / f"pred_{spec['label'].replace('=', '_')}.csv", index=False)
                rows.append(
                    {
                        "repeat": repeat + 1,
                        "m": m,
                        "method": spec["method"],
                        "variant": spec["label"],
                        **summarize_predictions(pred),
                    }
                )
                if "correction_gate" in pred.columns:
                    gate_rows.append(
                        {
                            "repeat": repeat + 1,
                            "m": m,
                            "variant": spec["label"],
                            "correction_gate_mean": float(pred["correction_gate"].mean()),
                            "correction_gate_std": float(pred["correction_gate"].std()),
                        }
                    )
                if trained.lambda_log is not None:
                    log = trained.lambda_log.copy()
                    log["repeat"] = repeat + 1
                    log["m"] = m
                    log["method"] = spec["method"]
                    log["variant"] = spec["label"]
                    lambda_rows.append(log)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = _summarize_repeated_metrics(metrics, out_dir / "metrics_summary.csv")
    if lambda_rows:
        pd.concat(lambda_rows, ignore_index=True).to_csv(out_dir / "lambda_logs.csv", index=False)
    if gate_rows:
        gates = pd.DataFrame(gate_rows)
        gates.to_csv(out_dir / "correction_gate.csv", index=False)
        gate_summary = gates.groupby(["m", "variant"], as_index=False).agg(
            correction_gate_mean=("correction_gate_mean", "mean"),
            correction_gate_std=("correction_gate_mean", "std"),
        )
        gate_summary.to_csv(out_dir / "correction_gate_summary.csv", index=False)
    else:
        gate_summary = pd.DataFrame()

    ranked = metrics.groupby(["m", "variant"], as_index=False).agg(rmse=("rmse", "mean"))
    ranked = ranked.sort_values(["m", "rmse"])
    lines = [
        "# DeepLesion Relative Anchored Correction Experiment",
        "",
        "- Target: relative log(V/V0)",
        "- Core idea: start from the latest observed CT value and learn a gated correction.",
        f"- Repeats: {repeats}",
        f"- Epochs: {train_cfg.epochs}",
        "",
        "## RMSE By Method",
        "",
        _markdown_table(ranked),
        "",
        "## Correction Gate Summary",
        "",
        _markdown_table(gate_summary) if not gate_summary.empty else "No gate rows.",
        "",
        f"Full metrics: `{summary}`",
        "",
    ]
    report = out_dir / "anchored_correction_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_deeplesion_state_space_uq_experiment(
    cfg: ExperimentConfig,
    epochs: int | None = None,
    repeats: int = 3,
) -> Path:
    """Evaluate measurement-noise-aware latent state-space UQ on relative DeepLesion."""
    from .state_space import predict_state_space, train_state_space_model
    from .train import train_single_model
    from .uq_methods import predict_deterministic, predict_mc_dropout

    out_dir = cfg.out_dir / "deeplesion_len5_relative_state_space_uq"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_deeplesion_len5_long(relative_log=True)
    ids = sorted(df["trajectory_id"].astype(str).unique().tolist())
    split = split_trajectory_ids_by_patient(
        df,
        ids,
        seed=cfg.cohort.seed + 205,
        test_fraction=cfg.cohort.test_fraction,
        val_fraction=cfg.cohort.val_fraction,
    )
    df.to_csv(out_dir / "cohort_deeplesion_len5_relative_long.csv", index=False)
    train_cfg = cfg.train if epochs is None else cfg.train.__class__(**{**cfg.train.__dict__, "epochs": epochs})
    specs = [
        {"method": "no_physics", "label": "plain_mlp", "cfg": {"fixed_lambda_phys": 0.0}, "stochastic": False},
        {"method": "fixed_pinn", "label": "fixed_gompertz_lambda=10", "cfg": {"fixed_lambda_phys": 10.0}, "stochastic": False},
        {"method": "adaptive_pinn", "label": "uncertainty_adaptive_beta=5", "cfg": {"adaptive_beta": 5.0}, "stochastic": True},
        {"method": "anchored_correction_pinn", "label": "anchored_correction_pinn_lambda=1", "cfg": {"fixed_lambda_phys": 1.0}, "stochastic": False},
    ]

    rows = []
    uncertainty_rows = []
    for repeat in range(repeats):
        repeat_seed = 134000 * repeat
        for m in [1, 2, 3, 4]:
            train_df = make_deeplesion_prediction_task(df, split.train, m=m)
            test_df = make_deeplesion_prediction_task(df, split.test, m=m)
            task_dir = out_dir / f"repeat{repeat + 1}" / f"m{m}"
            task_dir.mkdir(parents=True, exist_ok=True)
            pd.concat(
                [train_df.assign(split="train"), test_df.assign(split="test")],
                ignore_index=True,
            ).to_csv(task_dir / "task_rows.csv", index=False)

            persistence = _predict_persistence(test_df)
            persistence.to_csv(task_dir / "pred_persistence_anchor.csv", index=False)
            rows.append({"repeat": repeat + 1, "m": m, "method": "persistence", "variant": "persistence_anchor", **summarize_predictions(persistence)})

            state_space = train_state_space_model(train_df, train_cfg, seed=repeat_seed + 900 + m)
            state_space.train_log.to_csv(task_dir / "state_space_train_log.csv", index=False)
            state_pred = predict_state_space(state_space, test_df)
            state_pred.to_csv(task_dir / "pred_state_space_uq.csv", index=False)
            rows.append({"repeat": repeat + 1, "m": m, "method": "state_space_uq", "variant": "latent_state_space_uq", **summarize_predictions(state_pred)})
            uncertainty_rows.append(
                {
                    "repeat": repeat + 1,
                    "m": m,
                    "variant": "latent_state_space_uq",
                    "process_std_mean": float(state_pred["process_std"].mean()),
                    "measurement_std_mean": float(state_pred["measurement_std"].mean()),
                    "process_to_measurement_ratio": float(state_pred["process_std"].mean() / max(state_pred["measurement_std"].mean(), 1e-8)),
                }
            )

            for spec_idx, spec in enumerate(specs):
                spec_cfg = train_cfg.__class__(**{**train_cfg.__dict__, **spec["cfg"]})
                trained = train_single_model(
                    train_df,
                    spec_cfg,
                    method=spec["method"],
                    seed=repeat_seed + spec_idx * 101 + m,
                )
                if spec["stochastic"]:
                    pred = predict_mc_dropout(trained, test_df, samples=spec_cfg.mc_samples)
                else:
                    pred = predict_deterministic(trained, test_df)
                pred.to_csv(task_dir / f"pred_{spec['label'].replace('=', '_')}.csv", index=False)
                rows.append(
                    {
                        "repeat": repeat + 1,
                        "m": m,
                        "method": spec["method"],
                        "variant": spec["label"],
                        **summarize_predictions(pred),
                    }
                )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = _summarize_repeated_metrics(metrics, out_dir / "metrics_summary.csv")
    uq_summary = pd.DataFrame(uncertainty_rows)
    uq_summary.to_csv(out_dir / "uncertainty_decomposition.csv", index=False)
    uq_grouped = uq_summary.groupby(["m", "variant"], as_index=False).agg(
        process_std_mean=("process_std_mean", "mean"),
        measurement_std_mean=("measurement_std_mean", "mean"),
        process_to_measurement_ratio=("process_to_measurement_ratio", "mean"),
    )
    uq_grouped.to_csv(out_dir / "uncertainty_decomposition_summary.csv", index=False)

    ranked = metrics.groupby(["m", "variant"], as_index=False).agg(rmse=("rmse", "mean"))
    ranked = ranked.sort_values(["m", "rmse"])
    lines = [
        "# DeepLesion Relative Latent State-Space UQ Experiment",
        "",
        "- Target: relative log(V/V0)",
        "- Core idea: latent progression state plus separate process and measurement uncertainty.",
        f"- Repeats: {repeats}",
        f"- Epochs: {train_cfg.epochs}",
        "",
        "## RMSE By Method",
        "",
        _markdown_table(ranked),
        "",
        "## Uncertainty Decomposition",
        "",
        _markdown_table(uq_grouped),
        "",
        f"Full metrics: `{summary}`",
        "",
    ]
    report = out_dir / "state_space_uq_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_followup_density_experiment(cfg: ExperimentConfig, epochs: int | None = None, repeats: int = 3) -> Path:
    """Evaluate whether more observed CT nodes improve reliability on semi-synthetic L=5/L=8 trajectories."""
    from .train import train_single_model
    from .uq_methods import predict_deterministic, predict_mc_dropout

    out_dir = cfg.out_dir / "followup_density_nlstt300"
    out_dir.mkdir(parents=True, exist_ok=True)
    real_df = load_nlstt_long()
    ids = balanced_sample_trajectories(real_df, n=cfg.cohort.hmc_n, seed=cfg.cohort.seed + cfg.cohort.hmc_n)
    split = split_trajectory_ids(ids, seed=cfg.cohort.seed + cfg.cohort.hmc_n)
    train_cfg = cfg.train if epochs is None else cfg.train.__class__(**{**cfg.train.__dict__, "epochs": epochs})

    rows = []
    for repeat in range(repeats):
        repeat_seed = 40000 * repeat
        for generator in ["gompertz", "logistic"]:
            for total_nodes in [5, 8]:
                dense = generate_dense_followup_dataset(
                    real_df,
                    ids=ids,
                    total_nodes=total_nodes,
                    generator=generator,
                    seed=repeat_seed + total_nodes,
                    noise_sd=0.05,
                )
                dense_path = out_dir / f"repeat{repeat + 1}_{generator}_L{total_nodes}_dense.csv"
                dense.to_csv(dense_path, index=False)
                k_values = [1, 2, 3, 4] if total_nodes == 5 else [1, 2, 3, 4, 7]
                for k in k_values:
                    task_dir = out_dir / f"repeat{repeat + 1}" / generator / f"L{total_nodes}_k{k}"
                    task_dir.mkdir(parents=True, exist_ok=True)
                    train_df = make_dense_prediction_task(dense, split.train, k=k)
                    test_df = make_dense_prediction_task(dense, split.test, k=k)
                    train_df.to_csv(task_dir / "train_task.csv", index=False)
                    test_df.to_csv(task_dir / "test_task.csv", index=False)

                    fixed = train_single_model(train_df, train_cfg, method="fixed_pinn", seed=repeat_seed + total_nodes * 100 + k)
                    adaptive = train_single_model(
                        train_df,
                        train_cfg,
                        method="adaptive_pinn",
                        seed=repeat_seed + total_nodes * 100 + k + 50,
                    )
                    preds = {
                        "fixed_pinn": predict_deterministic(fixed, test_df),
                        "adaptive_pinn": predict_mc_dropout(adaptive, test_df, samples=train_cfg.mc_samples),
                    }
                    for method, pred in preds.items():
                        pred.to_csv(task_dir / f"pred_{method}.csv", index=False)
                        rows.append(
                            {
                                "repeat": repeat + 1,
                                "generator": generator,
                                "total_nodes": total_nodes,
                                "k": k,
                                "m": k,
                                "method": method,
                                "variant": f"{generator}_L{total_nodes}_k{k}",
                                **summarize_predictions(pred),
                            }
                        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    group_cols = ["generator", "total_nodes", "k", "method"]
    summary = (
        metrics.groupby(group_cols, as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            picp_mean=("picp", "mean"),
            picp_std=("picp", "std"),
            mpiw_mean=("mpiw", "mean"),
            mpiw_std=("mpiw", "std"),
        )
    )
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    return out_dir / "metrics_summary.csv"


def run_mechanistic_density_experiment(cfg: ExperimentConfig, repeats: int = 5) -> Path:
    """Mechanistic NLS follow-up density experiment for RQ4."""
    from .mechanistic_baselines import population_growth_prior, predict_density_mechanistic

    out_dir = cfg.out_dir / "mechanistic_density_nlstt300"
    out_dir.mkdir(parents=True, exist_ok=True)
    real_df = load_nlstt_long()
    ids = balanced_sample_trajectories(real_df, n=cfg.cohort.hmc_n, seed=cfg.cohort.seed + cfg.cohort.hmc_n)
    split = split_trajectory_ids(ids, seed=cfg.cohort.seed + cfg.cohort.hmc_n)

    rows = []
    for repeat in range(repeats):
        repeat_seed = 50000 * repeat
        for generator in ["gompertz", "logistic"]:
            for total_nodes in [5, 8]:
                dense = generate_dense_followup_dataset(
                    real_df,
                    ids=ids,
                    total_nodes=total_nodes,
                    generator=generator,
                    seed=repeat_seed + total_nodes,
                    noise_sd=0.04,
                )
                dense.to_csv(out_dir / f"repeat{repeat + 1}_{generator}_L{total_nodes}_dense.csv", index=False)
                prior = population_growth_prior(dense, split.train)
                k_values = [1, 2, 3, 4] if total_nodes == 5 else [1, 2, 3, 4, 7]
                fit_models = ["gompertz", "logistic", "log_linear"]
                for k in k_values:
                    for fit_model in fit_models:
                        pred = predict_density_mechanistic(dense, split.test, k=k, model=fit_model, prior_slope=prior)
                        pred.to_csv(out_dir / f"repeat{repeat + 1}_{generator}_L{total_nodes}_k{k}_{fit_model}.csv", index=False)
                        rows.append(
                            {
                                "repeat": repeat + 1,
                                "generator": generator,
                                "total_nodes": total_nodes,
                                "k": k,
                                "m": k,
                                "method": f"nls_{fit_model}",
                                "variant": f"{generator}_L{total_nodes}_k{k}_{fit_model}",
                                **summarize_predictions(pred),
                            }
                        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = (
        metrics.groupby(["generator", "total_nodes", "k", "method"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
        )
    )
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    best = summary.loc[summary.groupby(["generator", "total_nodes", "k"])["rmse_mean"].idxmin()].copy()
    best.to_csv(out_dir / "metrics_best_by_k.csv", index=False)
    return out_dir / "metrics_best_by_k.csv"


def write_experiment_plan(cfg: ExperimentConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "main_real_cohort": "NLSTt-500",
        "high_cost_uq_cohort": "NLSTt-300 for HMC/full Bayesian",
        "robustness_cohort": "NLSTt-1000 for low-cost methods",
        "real_sparse_tasks": {
            "m=1": "T0 -> T2",
            "m=2": "T0,T1 -> T2",
            "m=3": "T0,T1,T2 trajectory fitting/reconstruction diagnostics",
        },
        "semi_synthetic_tasks": {
            "m=5": "generated from real three-point NLSTt anchors",
            "m=8": "generated from real three-point NLSTt anchors",
        },
        "methods": [
            "no_physics",
            "fixed_pinn",
            "mc_dropout_pinn",
            "deep_ensemble_pinn",
            "variational_pinn",
            "laplace_pinn",
            "hmc_pinn on NLSTt-300",
            "adaptive_physics_weighting_pinn",
        ],
        "metrics": ["MAE", "RMSE", "MAPE", "PICP", "MPIW", "ECE-like calibration", "physics residual error"],
    }
    path = cfg.out_dir / "experiment_plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path
