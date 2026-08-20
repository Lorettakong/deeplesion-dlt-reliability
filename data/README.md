# Data Access and Expected Inputs

Raw DeepLesion CT images, DeepLesion annotations, and DeepLesion Tracking (DLT)
matching files are third-party resources and are not redistributed here.

To reproduce the complete benchmark construction pipeline, obtain the official
DeepLesion and DLT releases, then prepare the trajectory-level files expected by
the scripts.

## Source Data Role

| Data component | Role in this study |
|---|---|
| DeepLesion annotations | RECIST-style long/short axes, lesion boxes, file names, patient/scan metadata |
| DLT matching pairs | Source-target lesion links used to construct same-lesion tracking graphs |
| LesaNet / metadata labels | Body-site labels used for chest/lung, abdomen/liver, and other subgroup analyses |

## Expected Derived Inputs

The local cohort-construction workflow used the following derived files:

```text
outputs_deeplesion_longitudinal/deeplesion_trajectory_summary.csv
outputs_deeplesion_longitudinal/deeplesion_len5_trajectory_labels.csv
```

The manuscript-ready five-visit cohort expected by the released experiment
scripts is:

```text
outputs_nlstt_adaptive_uq_paper/deeplesion_len5_relative_main205/cohort_deeplesion_len5_long.csv
```

This file should contain one row per lesion observation with at least:

| Column type | Description |
|---|---|
| trajectory identifier | Same-lesion trajectory ID |
| patient identifier | Patient-level split key |
| visit index | Ordered CT visit index |
| lesion-size proxy | RECIST-based ellipsoid volume proxy or relative log-volume |
| body-site label | Anatomical subgroup label |

Column names are resolved by the loading utilities in
`nlstt_adaptive_uq_experiments/data.py`.

## Privacy and Redistribution

Do not commit raw CT images, downloaded DeepLesion/DLT archives, or large
intermediate image files. The repository only includes compact derived CSV
summaries and figure source data needed to inspect and regenerate manuscript
tables/figures.
