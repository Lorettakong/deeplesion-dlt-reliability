# Experiment 2: Refined UQ Method Comparison

## Code changes

- Deterministic raw predictions no longer create zero-width intervals. Raw PICP, MPIW, ECE, and NLL are reported as N/A because deterministic MLP has no native uncertainty estimator.
- Deep Ensemble now uses heteroscedastic members. Each ensemble member predicts a mean and log-variance, and the predictive variance combines aleatoric and epistemic components:
  `Var(y|x) = mean_k sigma_k^2 + Var_k(mu_k)`.
- Validation calibration uses scale conformal calibration for methods with native intervals:
  `s_i = |y_i - mu_i| / (sigma_i + eps)`, then `I(x) = [mu(x)-q sigma(x), mu(x)+q sigma(x)]`.
- Deterministic calibration uses validation absolute residual quantiles.
- MC Dropout sensitivity is reported for dropout rates 0.05, 0.10, 0.20, and 0.30 and stochastic forward passes 30, 50, and 100.

## Table 2A. Test RMSE

| m | Deterministic | MC Dropout | Deep Ensemble | Residual Gaussian |
| --- | --- | --- | --- | --- |
| 1 | 0.9662 +/- 0.0010 | 0.9646 +/- 0.0010 | 0.9646 +/- 0.0030 | 0.9644 +/- 0.0023 |
| 2 | 0.7637 +/- 0.0052 | 0.7578 +/- 0.0053 | 0.7476 +/- 0.0019 | 0.7649 +/- 0.0054 |
| 3 | 0.5318 +/- 0.0116 | 0.5327 +/- 0.0044 | 0.5361 +/- 0.0042 | 0.5409 +/- 0.0066 |
| 4 | 0.4390 +/- 0.0196 | 0.4253 +/- 0.0038 | 0.4275 +/- 0.0038 | 0.4275 +/- 0.0032 |

## Table 2B. m=4 Raw and Calibrated UQ Metrics

| Variant | Method | PICP | MPIW | ECE | NLL |
| --- | --- | --- | --- | --- | --- |
| raw | Deterministic | N/A | N/A | N/A | N/A |
| raw | MC Dropout | 0.2895 +/- 0.0788 | 0.3424 +/- 0.0253 | 0.5057 +/- 0.0226 | 14.2427 +/- 2.4466 |
| raw | Deep Ensemble | 0.9825 +/- 0.0172 | 2.3335 +/- 0.0520 | 0.0985 +/- 0.0077 | 0.6621 +/- 0.0152 |
| raw | Residual Gaussian | 0.9825 +/- 0.0172 | 2.4214 +/- 0.0177 | 0.1172 +/- 0.0094 | 0.6767 +/- 0.0057 |
| calibrated | Deterministic | 1.0000 +/- 0.0000 | 2.8381 +/- 0.1891 | 0.1468 +/- 0.0352 | 0.7812 +/- 0.0271 |
| calibrated | MC Dropout | 1.0000 +/- 0.0000 | 3.0927 +/- 0.1092 | 0.1522 +/- 0.0074 | 0.8233 +/- 0.0040 |
| calibrated | Deep Ensemble | 1.0000 +/- 0.0000 | 2.8170 +/- 0.0613 | 0.1566 +/- 0.0043 | 0.7668 +/- 0.0159 |
| calibrated | Residual Gaussian | 1.0000 +/- 0.0000 | 2.9323 +/- 0.1925 | 0.1676 +/- 0.0264 | 0.7921 +/- 0.0416 |

## Interpretation

Test RMSE is similar across UQ methods, indicating that the main differences among methods are not in point prediction means. The differences appear in raw coverage, interval width, ECE, and NLL.

MC Dropout remains under-covered in raw form even when dropout rate and sample count are varied, so validation calibration is necessary before using its intervals. Heteroscedastic Deep Ensemble and Residual Gaussian produce conservative raw intervals with high coverage and lower NLL. After validation calibration, all methods reach high coverage on the small test set, so PICP alone is insufficient. MPIW, ECE, and NLL are needed to evaluate whether the intervals are practically useful.
