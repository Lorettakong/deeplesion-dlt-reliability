# Experiment 5: Calibration Diagnostics

This diagnostic experiment uses the m=4 calibrated predictions from Experiment 2.

## Table 5A. m=4 diagnostic summary

| Method | PICP | MPIW | Uncertainty-error corr | Mean absolute error |
| --- | --- | --- | --- | --- |
| Deterministic | 1.0000 +/- 0.0000 | 2.8381 +/- 0.1671 | nan +/- nan | 0.3579 +/- 0.0302 |
| MC Dropout | 1.0000 +/- 0.0000 | 3.0927 +/- 0.0965 | -0.1326 +/- 0.0767 | 0.3387 +/- 0.0035 |
| Deep Ensemble | 1.0000 +/- 0.0000 | 2.8170 +/- 0.0542 | -0.0697 +/- 0.0071 | 0.3447 +/- 0.0020 |
| Residual Gaussian | 1.0000 +/- 0.0000 | 2.9323 +/- 0.1701 | nan +/- nan | 0.3378 +/- 0.0148 |

## Table 5B. Subgroup calibration

| Method | Subgroup | analysis | n_test_trajectories | covered_trajectories | PICP | PICP Clopper-Pearson 95% CI | MPIW | MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deterministic | abdomen/liver | exploratory | 13 | 13 | 1.0 | 0.7529-1.0000 | 2.8381 | 0.2718 |
| Deterministic | chest/lung | exploratory | 17 | 17 | 1.0 | 0.8049-1.0000 | 2.8381 | 0.3809 |
| Deterministic | other | exploratory | 8 | 8 | 1.0 | 0.6306-1.0000 | 2.8381 | 0.4488 |
| MC Dropout | abdomen/liver | exploratory | 13 | 13 | 1.0 | 0.7529-1.0000 | 3.099 | 0.2476 |
| MC Dropout | chest/lung | exploratory | 17 | 17 | 1.0 | 0.8049-1.0000 | 3.2153 | 0.3711 |
| MC Dropout | other | exploratory | 8 | 8 | 1.0 | 0.6306-1.0000 | 2.8218 | 0.4178 |
| Deep Ensemble | abdomen/liver | exploratory | 13 | 13 | 1.0 | 0.7529-1.0000 | 2.7934 | 0.2591 |
| Deep Ensemble | chest/lung | exploratory | 17 | 17 | 1.0 | 0.8049-1.0000 | 2.8621 | 0.3726 |
| Deep Ensemble | other | exploratory | 8 | 8 | 1.0 | 0.6306-1.0000 | 2.7597 | 0.4245 |
| Residual Gaussian | abdomen/liver | exploratory | 13 | 13 | 1.0 | 0.7529-1.0000 | 2.9323 | 0.2527 |
| Residual Gaussian | chest/lung | exploratory | 17 | 17 | 1.0 | 0.8049-1.0000 | 2.9323 | 0.359 |
| Residual Gaussian | other | exploratory | 8 | 8 | 1.0 | 0.6306-1.0000 | 2.9323 | 0.4311 |

## Table 5C. Coverage by target magnitude

| Method | Target magnitude | analysis | n_test_trajectories | covered_trajectories | PICP | PICP Clopper-Pearson 95% CI | MPIW | MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deterministic | middle y(T4) | exploratory | 13 | 13 | 1.0 | 0.7529-1.0000 | 2.8381 | 0.2607 |
| Deterministic | high y(T4) | exploratory | 12 | 12 | 1.0 | 0.7354-1.0000 | 2.8381 | 0.4216 |
| Deterministic | low y(T4) | exploratory | 13 | 13 | 1.0 | 0.7529-1.0000 | 2.8381 | 0.3963 |
| MC Dropout | middle y(T4) | exploratory | 13 | 13 | 1.0 | 0.7529-1.0000 | 2.6589 | 0.2322 |
| MC Dropout | high y(T4) | exploratory | 12 | 12 | 1.0 | 0.7354-1.0000 | 3.236 | 0.3794 |
| MC Dropout | low y(T4) | exploratory | 13 | 13 | 1.0 | 0.7529-1.0000 | 3.3941 | 0.4076 |
| Deep Ensemble | middle y(T4) | exploratory | 13 | 13 | 1.0 | 0.7529-1.0000 | 2.7336 | 0.2492 |
| Deep Ensemble | high y(T4) | exploratory | 12 | 12 | 1.0 | 0.7354-1.0000 | 2.8808 | 0.3749 |
| Deep Ensemble | low y(T4) | exploratory | 13 | 13 | 1.0 | 0.7529-1.0000 | 2.8417 | 0.4124 |
| Residual Gaussian | middle y(T4) | exploratory | 13 | 13 | 1.0 | 0.7529-1.0000 | 2.9323 | 0.2434 |
| Residual Gaussian | high y(T4) | exploratory | 12 | 12 | 1.0 | 0.7354-1.0000 | 2.9323 | 0.3518 |
| Residual Gaussian | low y(T4) | exploratory | 13 | 13 | 1.0 | 0.7529-1.0000 | 2.9323 | 0.4193 |

Figure: `outputs_nlstt_adaptive_uq_paper/experiment5_calibration_diagnostics/fig_experiment5_calibration_diagnostics.png`
