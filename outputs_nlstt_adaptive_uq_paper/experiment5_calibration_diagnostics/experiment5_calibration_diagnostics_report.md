# Experiment 5: Calibration Diagnostics

This diagnostic experiment uses the m=4 calibrated predictions from Experiment 2.
Subgroup and target-magnitude coverage intervals are exact two-sided Clopper-Pearson 95% confidence intervals computed from one independent test-set prediction per trajectory; repeated model fits are not counted as additional patients.

## Table 5A. m=4 diagnostic summary

| Method | PICP | MPIW | Uncertainty-error corr | Mean absolute error |
| --- | --- | --- | --- | --- |
| Deterministic | 1.0000 +/- 0.0000 | 2.9337 +/- 0.1523 | nan +/- nan | 0.3538 +/- 0.0178 |
| Cohort-level Feature GP | 1.0000 +/- 0.0000 | 2.9523 +/- 0.0000 | 0.0396 +/- 0.0000 | 0.3509 +/- 0.0000 |
| MC Dropout | 1.0000 +/- 0.0000 | 2.9512 +/- 0.1219 | -0.0285 +/- 0.0788 | 0.3387 +/- 0.0144 |
| Deep Ensemble | 1.0000 +/- 0.0000 | 2.8092 +/- 0.0372 | -0.0566 +/- 0.0204 | 0.3455 +/- 0.0036 |
| Gaussian residual-scale | 1.0000 +/- 0.0000 | 2.9318 +/- 0.1910 | nan +/- nan | 0.3504 +/- 0.0200 |

## Table 5B. Subgroup calibration

| Method | Subgroup | analysis | n_test_trajectories | covered_trajectories | Coverage fraction | PICP | PICP Clopper-Pearson 95% CI | MPIW | MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deterministic | abdomen/liver | exploratory | 13 | 13 | 13/13 | 1.0 | 0.7529-1.0000 | 2.9337 | 0.2614 |
| Deterministic | chest/lung | exploratory | 17 | 17 | 17/17 | 1.0 | 0.8049-1.0000 | 2.9337 | 0.3842 |
| Deterministic | other | exploratory | 8 | 8 | 8/8 | 1.0 | 0.6306-1.0000 | 2.9337 | 0.4396 |
| Cohort-level Feature GP | abdomen/liver | exploratory | 13 | 13 | 13/13 | 1.0 | 0.7529-1.0000 | 2.9509 | 0.2536 |
| Cohort-level Feature GP | chest/lung | exploratory | 17 | 17 | 17/17 | 1.0 | 0.8049-1.0000 | 2.9545 | 0.3801 |
| Cohort-level Feature GP | other | exploratory | 8 | 8 | 8/8 | 1.0 | 0.6306-1.0000 | 2.95 | 0.4469 |
| MC Dropout | abdomen/liver | exploratory | 13 | 13 | 13/13 | 1.0 | 0.7529-1.0000 | 2.9168 | 0.2477 |
| MC Dropout | chest/lung | exploratory | 17 | 17 | 17/17 | 1.0 | 0.8049-1.0000 | 3.0073 | 0.3744 |
| MC Dropout | other | exploratory | 8 | 8 | 8/8 | 1.0 | 0.6306-1.0000 | 2.8879 | 0.4109 |
| Deep Ensemble | abdomen/liver | exploratory | 13 | 13 | 13/13 | 1.0 | 0.7529-1.0000 | 2.7849 | 0.2595 |
| Deep Ensemble | chest/lung | exploratory | 17 | 17 | 17/17 | 1.0 | 0.8049-1.0000 | 2.8533 | 0.3737 |
| Deep Ensemble | other | exploratory | 8 | 8 | 8/8 | 1.0 | 0.6306-1.0000 | 2.7547 | 0.4253 |
| Gaussian residual-scale | abdomen/liver | exploratory | 13 | 13 | 13/13 | 1.0 | 0.7529-1.0000 | 2.9318 | 0.2594 |
| Gaussian residual-scale | chest/lung | exploratory | 17 | 17 | 17/17 | 1.0 | 0.8049-1.0000 | 2.9318 | 0.3787 |
| Gaussian residual-scale | other | exploratory | 8 | 8 | 8/8 | 1.0 | 0.6306-1.0000 | 2.9318 | 0.438 |

## Table 5C. Coverage by target magnitude

| Method | Target magnitude | analysis | n_test_trajectories | covered_trajectories | Coverage fraction | PICP | PICP Clopper-Pearson 95% CI | MPIW | MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deterministic | middle y(T4) | exploratory | 13 | 13 | 13/13 | 1.0 | 0.7529-1.0000 | 2.9337 | 0.2414 |
| Deterministic | high y(T4) | exploratory | 12 | 12 | 12/12 | 1.0 | 0.7354-1.0000 | 2.9337 | 0.4066 |
| Deterministic | low y(T4) | exploratory | 13 | 13 | 13/13 | 1.0 | 0.7529-1.0000 | 2.9337 | 0.4176 |
| Cohort-level Feature GP | middle y(T4) | exploratory | 14 | 14 | 14/14 | 1.0 | 0.7684-1.0000 | 2.9483 | 0.2607 |
| Cohort-level Feature GP | high y(T4) | exploratory | 12 | 12 | 12/12 | 1.0 | 0.7354-1.0000 | 2.9529 | 0.388 |
| Cohort-level Feature GP | low y(T4) | exploratory | 12 | 12 | 12/12 | 1.0 | 0.7354-1.0000 | 2.9564 | 0.4189 |
| MC Dropout | middle y(T4) | exploratory | 13 | 13 | 13/13 | 1.0 | 0.7529-1.0000 | 2.8417 | 0.2317 |
| MC Dropout | high y(T4) | exploratory | 12 | 12 | 12/12 | 1.0 | 0.7354-1.0000 | 3.0501 | 0.3557 |
| MC Dropout | low y(T4) | exploratory | 13 | 13 | 13/13 | 1.0 | 0.7529-1.0000 | 2.9693 | 0.43 |
| Deep Ensemble | middle y(T4) | exploratory | 13 | 13 | 13/13 | 1.0 | 0.7529-1.0000 | 2.7233 | 0.2497 |
| Deep Ensemble | high y(T4) | exploratory | 12 | 12 | 12/12 | 1.0 | 0.7354-1.0000 | 2.8771 | 0.3782 |
| Deep Ensemble | low y(T4) | exploratory | 13 | 13 | 13/13 | 1.0 | 0.7529-1.0000 | 2.8323 | 0.4112 |
| Gaussian residual-scale | middle y(T4) | exploratory | 13 | 13 | 13/13 | 1.0 | 0.7529-1.0000 | 2.9318 | 0.242 |
| Gaussian residual-scale | high y(T4) | exploratory | 12 | 12 | 12/12 | 1.0 | 0.7354-1.0000 | 2.9318 | 0.3964 |
| Gaussian residual-scale | low y(T4) | exploratory | 13 | 13 | 13/13 | 1.0 | 0.7529-1.0000 | 2.9318 | 0.4162 |

Figure: `outputs_nlstt_adaptive_uq_paper/experiment5_calibration_diagnostics/fig_experiment5_calibration_diagnostics.png`
