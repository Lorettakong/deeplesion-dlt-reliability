# DeepLesion-DLT Reliability Benchmark

This repository contains the code and release artifacts for the manuscript:

**Reliability Evaluation of Uncertainty Calibration and Gompertz-Inspired
Regularization in Sparse Longitudinal CT Lesion Prediction**

The study builds an automatically curated five-visit same-lesion CT benchmark
from DeepLesion and DeepLesion Tracking (DLT), then evaluates sparse-to-final
lesion-size prediction from a reliability-centered perspective: point
prediction accuracy, uncertainty calibration, interval sharpness, subgroup
behavior, statistical uncertainty, and trajectory regularization diagnostics.

## Repository Structure

```text
.
├── nlstt_adaptive_uq_experiments/     # Core Python package
│   ├── data.py                        # Cohort loading, task construction, split utilities
│   ├── models.py                      # Sparse trajectory MLP and model heads
│   ├── train.py                       # Training loops
│   ├── uq_methods.py                  # Deterministic, GP, MC Dropout, ensemble, residual-scale UQ
│   ├── metrics.py                     # RMSE, MAE, PICP, MPIW, ECE, NLL, calibration
│   ├── physics.py                     # Gompertz-inspired trajectory regularization
│   └── mechanistic_baselines.py       # Traditional longitudinal baselines
├── data_processing/                   # Reviewer-facing map for cohort construction/QC code
├── models/                            # Reviewer-facing map for prediction/UQ model code
├── experiments/                       # Reviewer-facing map for Exp0-Exp5 entry points
├── calibration/                       # Reviewer-facing map for calibration and UQ metrics
├── statistics/                        # Reviewer-facing map for bootstrap/CI analyses
├── figures/                           # Reviewer-facing map for manuscript figure regeneration
├── scripts/                           # Experiment entry points
├── outputs_nlstt_adaptive_uq_paper/   # Manuscript tables, source CSVs, and polished figures
├── data/README.md                     # Data access and expected input files
├── requirements.txt                   # Python dependencies
└── reproduce_figures.sh               # Regenerate polished manuscript figures from released CSVs
```

## Data Requirements

Raw DeepLesion CT images and DeepLesion Tracking files are third-party
resources and are **not redistributed** in this repository. To rerun the full
pipeline from source data, obtain DeepLesion and DLT from their official
releases and follow the notes in `data/README.md`.

The repository includes compact derived CSV outputs under
`outputs_nlstt_adaptive_uq_paper/` so that manuscript tables and figures can be
inspected and regenerated without redistributing raw CT images.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The editable installation makes the local package available to every script
while keeping code changes immediately visible. The experiments were
implemented in Python/PyTorch, with figure polishing
performed using R scripts saved beside the corresponding figure source data.

## Main Experimental Components

| Manuscript experiment | Purpose | Main script / outputs |
|---|---|---|
| Exp0 Data quality audit | Audit trajectory lengths, body-site distribution, patient split, and outliers | `scripts/build_experiment0_data_audit.py`, `scripts/plot_experiment0_data_audit.py` |
| Exp1 Follow-up density | Test whether more observed visits improve final-visit prediction; subgroup RMSE | `scripts/plot_experiment1_followup_density.py` |
| Exp1 traditional baselines | Compare neural/UQ models with classical longitudinal baselines | `scripts/run_experiment1_traditional_baselines.py` |
| Exp2 UQ comparison | Compare Deterministic MLP, Gaussian Process, MC Dropout, Deep Ensemble, and Gaussian residual-scale | `scripts/run_experiment2_uq_refined.py` |
| Exp2 calibration sensitivity | Compare 90% and 95% calibrated intervals | `scripts/run_experiment2_calibration_sensitivity.py` |
| Exp3 regularization weight | Evaluate Gompertz-inspired regularization weights | `scripts/run_experiment3_physics_refined.py` |
| Measurement-proxy sensitivity | Repeat the regularization comparison for ellipsoid volume, RECIST area, and long-axis length | `scripts/run_measurement_proxy_sensitivity.py` |
| Exp4 regularized UQ | Evaluate regularization with calibrated MC Dropout UQ | `scripts/run_experiment4_physics_uq_refined.py` |
| Exp5 diagnostics | Calibration, subgroup, target-magnitude, and interval diagnostics | `scripts/run_experiment5_calibration_diagnostics.py` |
| Statistical inference | Patient-level cluster bootstrap and paired differences | `scripts/run_patient_cluster_bootstrap_analysis.py` |
| Window sensitivity | Earliest/latest/sliding/random five-visit window sensitivity | `scripts/run_trajectory_window_sensitivity.py` |

## Model and Training Settings

The released scripts use the experimental settings described in the manuscript:

| Setting | Value |
|---|---|
| Model backbone | Sparse trajectory MLP encoder |
| Hidden layers | 2 |
| Hidden dimension | 64 |
| Activation | SiLU |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Batch size | 128 |
| Dropout rate | 0.10 for stochastic UQ models |
| MC Dropout samples | 50 stochastic forward passes |
| Deep Ensemble size | 5 independently initialized models |
| Calibration target | 95% prediction interval, alpha = 0.05 |
| Split rule | Patient-level development/calibration/test split (140/27/38 trajectories) |
| Main training repeats | 10 independent seeds |

Experiment-specific repeat counts, epochs, and sensitivity settings are encoded
in the corresponding scripts and output summaries.

## Reproducing Manuscript Figures

To regenerate polished PDF/PNG figures from the released derived CSV files:

```bash
bash reproduce_figures.sh
```

This runs the R figure scripts in:

- `outputs_nlstt_adaptive_uq_paper/experiment0_data_quality_audit/`
- `outputs_nlstt_adaptive_uq_paper/experiment1_followup_density/`
- `outputs_nlstt_adaptive_uq_paper/experiment2_uq_refined/`
- `outputs_nlstt_adaptive_uq_paper/experiment2_calibration_sensitivity/`
- `outputs_nlstt_adaptive_uq_paper/experiment3_physics_refined/`
- `outputs_nlstt_adaptive_uq_paper/experiment4_physics_uq_refined/`
- `outputs_nlstt_adaptive_uq_paper/experiment5_calibration_diagnostics/`

## Reproducing Full Experiments

Full model reruns require the derived five-visit DeepLesion-DLT cohort file
described in `data/README.md`. Once the cohort file is available, the main
experiment scripts can be run from the repository root, for example:

```bash
python scripts/run_experiment2_uq_refined.py
python scripts/run_experiment3_physics_refined.py
python scripts/run_measurement_proxy_sensitivity.py
python scripts/run_experiment4_physics_uq_refined.py
python scripts/run_experiment5_calibration_diagnostics.py
```

The scripts write manuscript-ready CSV summaries and figures into
`outputs_nlstt_adaptive_uq_paper/`.

## Citation

Please cite the associated manuscript when using this code. A BibTeX entry will
be added after publication.

## License

License information should be finalized before archival release.
