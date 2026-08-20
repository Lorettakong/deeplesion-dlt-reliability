# Statistical Analysis

This directory maps the manuscript statistical reporting to the implementation.

| Analysis | Implementation / output |
|---|---|
| Patient-level cluster bootstrap | `scripts/run_patient_cluster_bootstrap_analysis.py` |
| Paired RMSE difference bootstrap | `scripts/run_patient_cluster_bootstrap_analysis.py` |
| Subgroup bootstrap summaries | `outputs_nlstt_adaptive_uq_paper/patient_cluster_bootstrap/` |
| Exact binomial subgroup/target-magnitude coverage intervals | `scripts/run_experiment5_calibration_diagnostics.py` |

The patient-level cluster bootstrap resamples test patients with replacement
and keeps all trajectories from a sampled patient together, preserving
within-patient dependence during uncertainty estimation.
