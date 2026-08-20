# Calibration and UQ Metrics

This directory maps the manuscript calibration and uncertainty metrics to the
implementation.

| Component | Implementation |
|---|---|
| Split-conformal calibration utilities | `nlstt_adaptive_uq_experiments/metrics.py` |
| PICP, MPIW, ECE, and NLL | `nlstt_adaptive_uq_experiments/metrics.py` |
| Raw and calibrated UQ method wrappers | `nlstt_adaptive_uq_experiments/uq_methods.py` |
| 90% versus 95% calibration sensitivity | `scripts/run_experiment2_calibration_sensitivity.py` |
| Calibration and subgroup diagnostics | `scripts/run_experiment5_calibration_diagnostics.py` |

Released calibration summaries and figure source CSVs are stored under
`outputs_nlstt_adaptive_uq_paper/experiment2_calibration_sensitivity/` and
`outputs_nlstt_adaptive_uq_paper/experiment5_calibration_diagnostics/`.
