# Experiment 2: uncertainty-method comparison

## Design

All methods use the fixed-horizon recent-history task: the target is T4, and
the input contains the most recent `m` pre-target visits. Development,
calibration, and test sets are separated at patient level. Neural models are
summarized over 10 independent training seeds. Gaussian Process is fit once
per history length because its optimizer is deterministic for the fixed split.

The comparison includes Deterministic MLP, Gaussian Process, MC Dropout, Deep
Ensemble, and Gaussian residual-scale. Patient-level conformal calibration is
fit only on the calibration set. The test set is held out until final
evaluation.

## Released tables

- `table2a_calibrated_rmse.csv`: test RMSE for all five methods and four
  history lengths.
- `table2b_m4_raw_vs_calibrated_uq.csv`: raw and calibrated test PICP, MPIW,
  ECE, NLL, interval score, and one-level WIS at `m=4`.
- `gaussian_process_kernel_audit.csv`: optimized GP kernel and log marginal
  likelihood for each history length.
- `experiment2_split_counts.csv`: patient and trajectory counts for the fixed
  development/calibration/test split.

## Interpretation

Point-prediction RMSE is similar across methods. The larger differences concern
the uncertainty estimates: raw coverage, interval width, ECE, NLL, and interval
score. Calibrated intervals reach complete coverage on this small fixed test
set, so coverage must be interpreted together with sharpness and proper scoring
rules rather than as a stand-alone ranking criterion.
