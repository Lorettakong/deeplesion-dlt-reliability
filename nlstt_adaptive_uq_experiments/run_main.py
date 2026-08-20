from __future__ import annotations

import argparse

from .config import ExperimentConfig
from .experiments import (
    prepare_outputs,
    run_deeplesion_subgroup_gompertz_experiment,
    run_deeplesion_len5_main_experiment,
    run_deeplesion_reliability_gated_experiment,
    run_deeplesion_mixture_physics_experiment,
    run_deeplesion_feature_fusion_experiment,
    run_deeplesion_anchored_correction_experiment,
    run_deeplesion_state_space_uq_experiment,
    run_calibration_experiment,
    run_fair_nlstt300_experiment,
    run_followup_density_experiment,
    run_lambda_sweep_experiment,
    run_mechanistic_density_experiment,
    run_rq3_adaptive_mechanism_experiment,
    run_rq2_physics_value_experiment,
    run_rq1_uq_comparison,
    run_smoke_experiment,
    write_experiment_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "plan",
            "prepare",
            "smoke",
            "fair",
            "calibrate",
            "lambda",
            "density",
            "density_nls",
            "rq1",
            "rq2",
            "rq3",
            "deeplesion",
            "deeplesion_relative",
            "deeplesion_subgroup_gompertz",
            "deeplesion_rg",
            "deeplesion_mixture",
            "deeplesion_fusion",
            "deeplesion_anchor",
            "deeplesion_state",
        ],
        default="plan",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.stage == "plan":
        path = write_experiment_plan(cfg)
        print(f"Wrote plan: {path}")
    elif args.stage == "prepare":
        prepare_outputs(cfg)
        path = write_experiment_plan(cfg)
        print(f"Wrote cohorts and plan under: {cfg.out_dir}")
        print(f"Plan: {path}")
    elif args.stage == "smoke":
        path = run_smoke_experiment(cfg)
        print(f"Smoke metrics: {path}")
    elif args.stage == "fair":
        path = run_fair_nlstt300_experiment(cfg, epochs=args.epochs, repeats=args.repeats)
        print(f"Fair NLSTt-300 metrics: {path}")
    elif args.stage == "calibrate":
        path = run_calibration_experiment(cfg, epochs=args.epochs, repeats=args.repeats)
        print(f"Calibration metrics: {path}")
    elif args.stage == "lambda":
        path = run_lambda_sweep_experiment(cfg, epochs=args.epochs, repeats=args.repeats)
        print(f"Lambda sweep metrics: {path}")
    elif args.stage == "density":
        path = run_followup_density_experiment(cfg, epochs=args.epochs, repeats=args.repeats)
        print(f"Follow-up density metrics: {path}")
    elif args.stage == "density_nls":
        path = run_mechanistic_density_experiment(cfg, repeats=args.repeats)
        print(f"Mechanistic density metrics: {path}")
    elif args.stage == "rq1":
        path = run_rq1_uq_comparison(cfg, epochs=args.epochs, repeats=args.repeats)
        print(f"RQ1 UQ comparison metrics: {path}")
    elif args.stage == "rq2":
        path = run_rq2_physics_value_experiment(cfg, epochs=args.epochs, repeats=args.repeats)
        print(f"RQ2 physics value metrics: {path}")
    elif args.stage == "rq3":
        path = run_rq3_adaptive_mechanism_experiment(cfg, epochs=args.epochs, repeats=args.repeats)
        print(f"RQ3 adaptive mechanism metrics: {path}")
    elif args.stage == "deeplesion":
        path = run_deeplesion_len5_main_experiment(cfg, epochs=args.epochs, repeats=args.repeats)
        print(f"DeepLesion len5 report: {path}")
    elif args.stage == "deeplesion_relative":
        path = run_deeplesion_len5_main_experiment(cfg, epochs=args.epochs, repeats=args.repeats, relative_log=True)
        print(f"DeepLesion relative len5 report: {path}")
    elif args.stage == "deeplesion_subgroup_gompertz":
        path = run_deeplesion_subgroup_gompertz_experiment(cfg)
        print(f"DeepLesion subgroup Gompertz report: {path}")
    elif args.stage == "deeplesion_rg":
        path = run_deeplesion_reliability_gated_experiment(cfg, epochs=args.epochs, repeats=args.repeats)
        print(f"DeepLesion RG-PINN-UQ report: {path}")
    elif args.stage == "deeplesion_mixture":
        path = run_deeplesion_mixture_physics_experiment(cfg, epochs=args.epochs, repeats=args.repeats)
        print(f"DeepLesion mixture physics report: {path}")
    elif args.stage == "deeplesion_fusion":
        path = run_deeplesion_feature_fusion_experiment(cfg, epochs=args.epochs, repeats=args.repeats)
        print(f"DeepLesion physics feature fusion report: {path}")
    elif args.stage == "deeplesion_anchor":
        path = run_deeplesion_anchored_correction_experiment(cfg, epochs=args.epochs, repeats=args.repeats)
        print(f"DeepLesion anchored correction report: {path}")
    elif args.stage == "deeplesion_state":
        path = run_deeplesion_state_space_uq_experiment(cfg, epochs=args.epochs, repeats=args.repeats)
        print(f"DeepLesion state-space UQ report: {path}")


if __name__ == "__main__":
    main()
