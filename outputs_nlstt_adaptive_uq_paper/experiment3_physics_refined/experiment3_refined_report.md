# Experiment 3: Fixed Physics-Weight Sensitivity

## Mechanism

The physics term is a last-observed-to-target discrete Gompertz residual on relative log-volume. For relative log-volume y(t)=log(V(t)/V(0)), the Gompertz form is dy/dt=a(c-y). Because the network predicts only the final visit, the implemented residual is r = (y_hat_T - y_last) / Delta t - a(c - 0.5*(y_last + y_hat_T)). Here a and c are trajectory-conditioned outputs of the network, and Delta t is the visit-index interval.

## Table 3A. RMSE

| m | No Regularization | lambda=0.1 | lambda=1 | lambda=10 | lambda=100 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.9652 +/- 0.0034 | 0.9656 +/- 0.0031 | 0.9652 +/- 0.0020 | 0.9664 +/- 0.0037 | 0.9644 +/- 0.0023 |
| 2 | 0.7588 +/- 0.0061 | 0.7652 +/- 0.0091 | 0.7589 +/- 0.0047 | 0.7675 +/- 0.0088 | 0.7585 +/- 0.0038 |
| 3 | 0.5422 +/- 0.0042 | 0.5418 +/- 0.0108 | 0.5405 +/- 0.0145 | 0.5365 +/- 0.0050 | 0.5369 +/- 0.0075 |
| 4 | 0.4408 +/- 0.0136 | 0.4295 +/- 0.0095 | 0.4331 +/- 0.0080 | 0.4307 +/- 0.0058 | 0.5450 +/- 0.0322 |

## Table 3B. Physics residual

| m | No Regularization | lambda=0.1 | lambda=1 | lambda=10 | lambda=100 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0415 +/- 0.0198 | 0.0052 +/- 0.0037 | 0.0096 +/- 0.0046 | 0.0043 +/- 0.0022 | 0.0013 +/- 0.0012 |
| 2 | 0.3259 +/- 0.0366 | 0.0076 +/- 0.0041 | 0.0068 +/- 0.0029 | 0.0162 +/- 0.0085 | 0.0071 +/- 0.0054 |
| 3 | 0.3130 +/- 0.0414 | 0.0234 +/- 0.0084 | 0.0296 +/- 0.0131 | 0.0144 +/- 0.0083 | 0.0059 +/- 0.0030 |
| 4 | 0.3623 +/- 0.0861 | 0.0457 +/- 0.0118 | 0.0400 +/- 0.0131 | 0.0178 +/- 0.0080 | 0.0080 +/- 0.0032 |

## Table 3C. Best lambda trade-off

| m | best_lambda_by_rmse | best_rmse | best_lambda_by_residual | best_residual |
| --- | --- | --- | --- | --- |
| 1 | lambda=100 | 0.9644 | lambda=100 | 0.0013 |
| 2 | lambda=100 | 0.7585 | lambda=1 | 0.0068 |
| 3 | lambda=10 | 0.5365 | lambda=100 | 0.0059 |
| 4 | lambda=0.1 | 0.4295 | lambda=100 | 0.0080 |

## Interpretation

Fixed physics regularization shows an accuracy-consistency trade-off. Nonzero lambda values consistently reduce the Gompertz residual relative to the no-physics model, but the lambda that minimizes residual is not necessarily the lambda that minimizes RMSE. Therefore, the physics term should be interpreted as consistency regularization rather than an unconditional accuracy booster.
