import unittest

import numpy as np
import pandas as pd

from nlstt_adaptive_uq_experiments.data import (
    make_deeplesion_prediction_task,
    patient_group_kfold_ids,
)


def _cohort() -> pd.DataFrame:
    rows = []
    for trajectory_index in range(10):
        for visit in range(5):
            rows.append(
                {
                    "trajectory_id": f"trajectory-{trajectory_index}",
                    "patient_id": f"patient-{trajectory_index}",
                    "growth_class": "stable",
                    "t_rel": float(visit),
                    "logV": float(10 * trajectory_index + visit),
                    "V_obs_cm3": float(np.exp(visit)),
                }
            )
    return pd.DataFrame(rows)


class FixedHorizonTaskTests(unittest.TestCase):
    def test_recent_history_always_targets_t4(self) -> None:
        cohort = _cohort()
        ids = cohort["trajectory_id"].drop_duplicates().tolist()
        for history_length in (1, 2, 3, 4):
            task = make_deeplesion_prediction_task(cohort, ids, history_length)
            self.assertEqual(len(task), len(ids))
            self.assertTrue((task["t_target"] == 4.0).all())
            self.assertTrue(task["t_obs"].map(len).eq(history_length).all())
            expected_times = list(np.arange(4 - history_length, 4, dtype=float))
            self.assertTrue(task["t_obs"].map(lambda value: value == expected_times).all())

    def test_patient_grouped_folds_are_disjoint(self) -> None:
        cohort = _cohort()
        ids = cohort["trajectory_id"].drop_duplicates().tolist()
        patient_by_trajectory = (
            cohort.groupby("trajectory_id")["patient_id"].first().astype(str).to_dict()
        )
        validation_ids = []
        for train_ids, fold_ids in patient_group_kfold_ids(cohort, ids, n_splits=5, seed=17):
            train_patients = {patient_by_trajectory[item] for item in train_ids}
            fold_patients = {patient_by_trajectory[item] for item in fold_ids}
            self.assertTrue(train_patients.isdisjoint(fold_patients))
            validation_ids.extend(fold_ids)
        self.assertEqual(sorted(validation_ids), sorted(ids))


if __name__ == "__main__":
    unittest.main()
