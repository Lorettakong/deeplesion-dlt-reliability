# Formal full rerun validation (2026-09-01)

## Scope

- Experiment 1: fixed deterministic MLP, recent-history windows, 10 training seeds.
- Experiment 2: Deterministic, Gaussian Process, MC Dropout, Deep Ensemble, and Gaussian residual-scale; 10 training seeds; patient-grouped development CV; patient-level conformal calibration.
- Experiment 3: five fixed physics weights, four history lengths, 10 training seeds.
- Experiment 4: lambda 1 and lambda 10, four history lengths, 10 training seeds.
- Measurement-proxy sensitivity: ellipsoid volume, RECIST area, and RECIST long axis; 10 seeds.
- Experiment 5: calibration diagnostics including explicit covered/evaluated subgroup counts.

## Integrity checks

- Development/calibration/test patients overlap: 0 for every pair of splits.
- Development/calibration/test trajectories: 140/27/38.
- Development/calibration/test patients: 84/19/26.
- Experiment 2 metric rows: 600; independent training seeds: 10.
- Experiment 3 metric rows: 200; independent training seeds: 10.
- Experiment 4 metric rows: 160 per lambda; independent training seeds: 10.
- Measurement-proxy sensitivity rows: 60; independent training seeds: 10.
- Gaussian-process kernel audit rows: 40; all log marginal likelihoods finite.
- Every prediction file contains exactly the fixed test trajectory set.
- All core point predictions and RMSE values are finite.
- Pooled/subgroup RMSE reconstruction maximum absolute discrepancy: < 1e-12.
- Current and strict-unambiguous cohorts are identical: 205 trajectories and 1,025 time points.

## Principal numerical checks

Experiment 1 deterministic RMSE (mean across seeds):

| m | RMSE | 95% CI half-width |
|---:|---:|---:|
| 1 | 0.468453 | 0.003880 |
| 2 | 0.444552 | 0.011540 |
| 3 | 0.433226 | 0.004772 |
| 4 | 0.440893 | 0.008211 |

Experiment 3 at m=4:

| lambda | RMSE | Mean absolute physics residual |
|---:|---:|---:|
| 0 | 0.437242 | 0.085882 |
| 0.1 | 0.437003 | 0.095945 |
| 1 | 0.445017 | 0.050580 |
| 10 | 0.460456 | 0.019760 |
| 100 | 0.463635 | 0.007268 |

The rerun supports a consistency--accuracy trade-off, not an unconditional accuracy gain from stronger physics regularization.

Measurement-proxy sensitivity consistently found higher RMSE but lower physics residual at lambda 1 relative to lambda 0 for all three proxies. This supports the same narrow trade-off interpretation across the prespecified RECIST-derived scales.

## Backup

Pre-rerun outputs were copied to `full_rerun_backup_20260901_2115` under the paper output directory.
