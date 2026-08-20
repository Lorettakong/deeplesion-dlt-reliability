# Experiment 0: Data Quality Audit and Benchmark Characterization

## Candidate Trajectory Audit
| criterion | trajectories |
| --- | --- |
| All DLT connected components | 2117 |
| Patient-consistent components | 2117 |
| Trajectory length >= 3 | 811 |
| Trajectory length = 4 | 195 |
| Trajectory length >= 5 | 205 |
| Trajectory length >= 8 | 43 |

## Main Cohort Overview
| quantity | value |
| --- | --- |
| Trajectories | 205 |
| Patients | 129 |
| Time points | 1025 |
| Unique scans per retained trajectory | 5 |

## Patient-Level Split
| split | patients | trajectories | purpose |
| --- | --- | --- | --- |
| train | 84 | 140 | Model fitting |
| validation | 19 | 27 | Model selection and calibration |
| test | 26 | 38 | Final evaluation only |

## Trajectories Per Patient
| trajectories_per_patient | patients |
| --- | --- |
| 1 | 83 |
| 2 | 27 |
| 3 | 10 |
| 4 | 7 |
| 5 | 2 |

## Body-Site Distribution
| body_region_group | trajectories | percent |
| --- | --- | --- |
| chest_or_lung | 90 | 43.9 |
| abdomen_or_liver | 70 | 34.1 |
| other_or_unknown | 19 | 9.3 |
| lymphatic | 14 | 6.8 |
| pelvis | 11 | 5.4 |
| soft_tissue_or_breast | 1 | 0.5 |

## Task Sample Counts by m
| m | split | patients | task_samples | input_visits | target_visit |
| --- | --- | --- | --- | --- | --- |
| 1 | train | 84 | 140 | 1 | T4 |
| 1 | validation | 19 | 27 | 1 | T4 |
| 1 | test | 26 | 38 | 1 | T4 |
| 2 | train | 84 | 140 | 2 | T4 |
| 2 | validation | 19 | 27 | 2 | T4 |
| 2 | test | 26 | 38 | 2 | T4 |
| 3 | train | 84 | 140 | 3 | T4 |
| 3 | validation | 19 | 27 | 3 | T4 |
| 3 | test | 26 | 38 | 3 | T4 |
| 4 | train | 84 | 140 | 4 | T4 |
| 4 | validation | 19 | 27 | 4 | T4 |
| 4 | test | 26 | 38 | 4 | T4 |

## Volume Variable Distribution
| variable | n | mean | sd | min | p25 | median | p75 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raw RECIST volume proxy V(t), cm3 | 1025 | 35.9186 | 97.5913 | 0.0105 | 1.9088 | 5.3798 | 21.9585 | 1135.9713 |
| log V(t) | 1025 | 1.8408 | 1.9075 | -4.554 | 0.6465 | 1.6826 | 3.0892 | 7.0352 |
| relative log-volume log(V(t)/V0) | 1025 | -0.004 | 0.8485 | -3.1594 | -0.3775 | 0.0 | 0.3953 | 4.7534 |

## Outlier Audit (IQR Rule)
| variable | q1 | q3 | iqr | lower_fence | upper_fence | outlier_n | outlier_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Raw RECIST volume proxy V(t), cm3 | 1.9088 | 21.9585 | 20.0497 | -28.1657 | 52.0331 | 153 | 14.9268 |
| log V(t) | 0.6465 | 3.0892 | 2.4427 | -3.0175 | 6.7532 | 9 | 0.878 |
| relative log-volume log(V(t)/V0) | -0.3775 | 0.3953 | 0.7729 | -1.5369 | 1.5547 | 87 | 8.4878 |
| T4 target Raw RECIST volume proxy V(t), cm3 | 1.8895 | 23.0984 | 21.2089 | -29.9238 | 54.9117 | 28 | 13.6585 |
| T4 target log V(t) | 0.6363 | 3.1398 | 2.5034 | -3.1188 | 6.8949 | 2 | 0.9756 |
| T4 target relative log-volume log(V(t)/V0) | -0.7078 | 0.7171 | 1.4249 | -2.8451 | 2.8544 | 5 | 2.439 |
