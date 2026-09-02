# Data Processing

This directory is a reviewer-facing guide to the cohort construction and
quality-control code used in the manuscript. The implementation lives in the
core package and experiment scripts rather than being duplicated here.

| Task | Implementation |
|---|---|
| Load trajectory-level DeepLesion-DLT inputs | `nlstt_adaptive_uq_experiments/data.py` |
| Build deterministic current/strict trajectory cohorts without future-size selection | `scripts/prepare_deeplesion_longitudinal.py` |
| Construct fixed-horizon recent-history prediction tables | `nlstt_adaptive_uq_experiments/data.py` |
| Build Experiment 0 cohort audit summaries | `scripts/build_experiment0_data_audit.py` |
| Plot Experiment 0 data quality audit | `scripts/plot_experiment0_data_audit.py` |
| Document expected raw and derived inputs | `data/README.md` |

Raw DeepLesion images and DLT files are not redistributed. See
`data/README.md` for the expected derived cohort files and privacy notes.
