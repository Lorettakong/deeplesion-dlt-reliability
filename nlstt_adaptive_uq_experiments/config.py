from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NLSTT_LONG_CSV = PROJECT_ROOT / "data" / "nlstt_processed" / "nlstt_three_scan_volume_long_with_metadata.csv"
NLSTT_ALL_ELIGIBLE_SUMMARY_CSV = (
    PROJECT_ROOT / "data" / "nlstt_processed" / "nlstt_three_scan_all_eligible_summary.csv"
)
OUT_DIR = PROJECT_ROOT / "outputs_nlstt_adaptive_uq_paper"
DEEPLESION_TRAJ_LEN5_CSV = PROJECT_ROOT / "outputs_deeplesion_longitudinal" / "deeplesion_trajectories_len5.csv"
DEEPLESION_TRAJ_LABELS_CSV = PROJECT_ROOT / "outputs_deeplesion_longitudinal" / "deeplesion_len5_trajectory_labels.csv"


@dataclass(frozen=True)
class CohortConfig:
    main_n: int = 500
    hmc_n: int = 300
    robustness_n: int = 1000
    seed: int = 20260624
    test_fraction: float = 0.20
    val_fraction: float = 0.15


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 800
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 64
    dropout: float = 0.10
    fixed_lambda_phys: float = 10.0
    adaptive_lambda_min: float = 0.1
    adaptive_lambda_max: float = 100.0
    adaptive_beta: float = 5.0
    reliability_uq_beta: float = 2.0
    reliability_phys_gamma: float = 1.0
    reliability_sparsity_power: float = 1.0
    mixture_entropy_weight: float = 0.01
    ensemble_size: int = 5
    mc_samples: int = 50


@dataclass(frozen=True)
class ExperimentConfig:
    cohort: CohortConfig = CohortConfig()
    train: TrainConfig = TrainConfig()
    out_dir: Path = OUT_DIR
