# Experiment 7: Time-Interval Sensitivity

Models: deterministic no-physics MLP; repeats: 3; epochs: 400.

## Table 7A. Time availability audit

| Item | Value | Interpretation |
| --- | --- | --- |
| Explicit scan date columns | None found | Real calendar time is unavailable in the current DeepLesion-DLT table |
| Trajectories with proxy time different from visit index | 64/205 | Only a subset can test proxy-time sensitivity |
| Trajectories with duplicated study_id among first five nodes | 46/205 | DLT components may contain multiple lesion nodes from the same study; this is not a true date interval |
| Mean max absolute difference between proxy time and visit index | 0.2374 | Proxy time differs mildly from visit index on average |

## Table 7B. Visit-index input versus StudyID proxy-time input

| m | Time input | RMSE | MAE |
| --- | --- | --- | --- |
| 1 | Visit index | 0.9670 +/- 0.0009 | 0.7750 +/- 0.0005 |
| 1 | StudyID proxy time | 0.9650 +/- 0.0020 | 0.7762 +/- 0.0012 |
| 2 | Visit index | 0.7747 +/- 0.0098 | 0.6021 +/- 0.0128 |
| 2 | StudyID proxy time | 0.7613 +/- 0.0056 | 0.5999 +/- 0.0072 |
| 3 | Visit index | 0.5339 +/- 0.0041 | 0.4155 +/- 0.0087 |
| 3 | StudyID proxy time | 0.5680 +/- 0.0090 | 0.4495 +/- 0.0173 |
| 4 | Visit index | 0.4362 +/- 0.0046 | 0.3458 +/- 0.0182 |
| 4 | StudyID proxy time | 0.4919 +/- 0.0165 | 0.3870 +/- 0.0166 |

Note: physics residual is not used as a primary endpoint in this experiment because StudyID proxy time can contain duplicated or nearly duplicated time values. This confirms that StudyID is not a valid substitute for real scan dates.

Figure: `outputs_nlstt_adaptive_uq_paper/experiment7_time_interval_sensitivity/fig_experiment7_time_interval_sensitivity.png`
