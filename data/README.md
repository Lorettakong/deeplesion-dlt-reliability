# Data

Raw DeepLesion and DeepLesion Tracking files are third-party resources and are
not redistributed in this repository.

To rerun the full pipeline, obtain the source datasets from their official
releases and prepare the trajectory-level inputs expected by the scripts. The
original local workflow used:

- `outputs_deeplesion_longitudinal/deeplesion_trajectory_summary.csv`
- `outputs_deeplesion_longitudinal/deeplesion_len5_trajectory_labels.csv`

The manuscript-ready derived cohort used by the released experiment scripts is:

- `outputs_nlstt_adaptive_uq_paper/deeplesion_len5_relative_main205/cohort_deeplesion_len5_long.csv`

Do not commit raw CT images or downloaded DeepLesion/DLT source files.
