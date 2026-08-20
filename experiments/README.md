# Experiments

This directory maps each manuscript experiment to the corresponding runnable
entry point or released output folder.

| Manuscript experiment | Script / output |
|---|---|
| Experiment 0: data quality audit | `scripts/build_experiment0_data_audit.py`; `outputs_nlstt_adaptive_uq_paper/experiment0_data_quality_audit/` |
| Experiment 1: follow-up density | `scripts/plot_experiment1_followup_density.py`; `outputs_nlstt_adaptive_uq_paper/experiment1_followup_density/` |
| Experiment 1: traditional baselines | `scripts/run_experiment1_traditional_baselines.py`; `outputs_nlstt_adaptive_uq_paper/experiment1_traditional_baselines/` |
| Experiment 2: UQ method comparison | `scripts/run_experiment2_uq_refined.py`; `outputs_nlstt_adaptive_uq_paper/experiment2_uq_refined/` |
| Experiment 2: calibration sensitivity | `scripts/run_experiment2_calibration_sensitivity.py`; `outputs_nlstt_adaptive_uq_paper/experiment2_calibration_sensitivity/` |
| Experiment 3: regularization-weight sensitivity | `scripts/run_experiment3_physics_refined.py`; `outputs_nlstt_adaptive_uq_paper/experiment3_physics_refined/` |
| Experiment 4: regularized UQ reliability | `scripts/run_experiment4_physics_uq_refined.py`; `outputs_nlstt_adaptive_uq_paper/experiment4_physics_uq_refined/` |
| Experiment 5: calibration and subgroup diagnostics | `scripts/run_experiment5_calibration_diagnostics.py`; `outputs_nlstt_adaptive_uq_paper/experiment5_calibration_diagnostics/` |
| Trajectory-window sensitivity | `scripts/run_trajectory_window_sensitivity.py`; `scripts/run_trajectory_window_sensitivity_mlp.py` |

Run `bash reproduce_figures.sh` from the repository root to regenerate the
polished manuscript figures from the released source CSV files.
