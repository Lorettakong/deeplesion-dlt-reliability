# DeepLesion-DLT Reliability Evaluation Code

This repository contains code and release artifacts for the manuscript
"Reliability Evaluation of Uncertainty Calibration and Gompertz-Inspired
Regularization in Sparse Longitudinal CT Lesion Prediction."

## Repository Structure

- `nlstt_adaptive_uq_experiments/`: core Python package for data preparation,
  model training, uncertainty prediction, calibration, and metrics.
- `scripts/`: experiment and figure-generation entry points used for the
  manuscript.
- `outputs_nlstt_adaptive_uq_paper/`: selected reproducibility outputs,
  manuscript tables, CSV summaries, and polished figure scripts/assets.
- `data/`: data-access notes. Raw DeepLesion and DeepLesion Tracking files are
  not redistributed.

## Data

Raw DeepLesion and DeepLesion Tracking data are third-party resources and are
not included in this repository. To rerun the full pipeline, obtain the source
datasets from their official releases and place processed trajectory inputs in
the locations described in `data/README.md`.

The included `outputs_nlstt_adaptive_uq_paper/` directory contains compact
derived outputs used to reproduce manuscript tables and figures.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example Commands

```bash
python scripts/run_experiment2_uq_refined.py
python scripts/run_experiment5_calibration_diagnostics.py
Rscript outputs_nlstt_adaptive_uq_paper/experiment2_uq_refined/plot_experiment2_uq_refined_cmpb.R
```

## Notes

The package name `nlstt_adaptive_uq_experiments` is retained as a legacy
internal module name. The public experiment scripts and outputs correspond to
the DeepLesion-DLT sparse longitudinal CT reliability benchmark.

## License and Citation

License and citation information should be updated before public release.
