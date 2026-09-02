# Models

This directory maps the manuscript model descriptions to their implementation
files.

| Method | Implementation |
|---|---|
| Sparse trajectory MLP backbone | `nlstt_adaptive_uq_experiments/models.py` |
| Deterministic MLP | `nlstt_adaptive_uq_experiments/uq_methods.py` |
| Gaussian Process | `nlstt_adaptive_uq_experiments/uq_methods.py` |
| MC Dropout | `nlstt_adaptive_uq_experiments/uq_methods.py` |
| Deep Ensemble | `nlstt_adaptive_uq_experiments/uq_methods.py` |
| Gaussian residual-scale baseline | `nlstt_adaptive_uq_experiments/uq_methods.py` |
| Gompertz-inspired trajectory regularization | `nlstt_adaptive_uq_experiments/physics.py` |
| Traditional longitudinal baselines | `nlstt_adaptive_uq_experiments/mechanistic_baselines.py` |
| Training loops | `nlstt_adaptive_uq_experiments/train.py` |

The top-level `README.md` summarizes the manuscript training settings,
including hidden dimension, optimizer, dropout, MC samples, and ensemble size.
