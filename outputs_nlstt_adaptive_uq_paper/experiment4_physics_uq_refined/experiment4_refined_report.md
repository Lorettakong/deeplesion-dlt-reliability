# Experiment 4: Gompertz-Inspired Regularization and UQ Reliability

Main setting: Gompertz-inspired regularization lambda=1 + MC Dropout UQ versus No Regularization + MC Dropout UQ.

## Table 4A. Lambda=1 calibrated UQ results

| m | Method | RMSE | PICP | MPIW | ECE | NLL | Gompertz-style residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | No Regularization + UQ | 0.9664 +/- 0.0016 | 1.0000 +/- 0.0000 | 6.3556 +/- 0.0491 | 0.1632 +/- 0.0019 | 1.5799 +/- 0.0055 | 0.0577 +/- 0.0376 |
| 1 | Gompertz-inspired regularization lambda=1 + UQ | 0.9640 +/- 0.0016 | 1.0000 +/- 0.0000 | 6.3410 +/- 0.0311 | 0.1621 +/- 0.0057 | 1.5775 +/- 0.0037 | 0.0034 +/- 0.0013 |
| 2 | No Regularization + UQ | 0.7806 +/- 0.0048 | 0.9825 +/- 0.0152 | 3.5456 +/- 0.2422 | 0.0631 +/- 0.0177 | 1.1885 +/- 0.0169 | 0.3509 +/- 0.0735 |
| 2 | Gompertz-inspired regularization lambda=1 + UQ | 0.7758 +/- 0.0041 | 0.9737 +/- 0.0000 | 3.4693 +/- 0.1448 | 0.0619 +/- 0.0041 | 1.1772 +/- 0.0098 | 0.0078 +/- 0.0051 |
| 3 | No Regularization + UQ | 0.5351 +/- 0.0114 | 1.0000 +/- 0.0000 | 3.2107 +/- 0.0796 | 0.1413 +/- 0.0076 | 0.9285 +/- 0.0201 | 0.4476 +/- 0.0623 |
| 3 | Gompertz-inspired regularization lambda=1 + UQ | 0.5374 +/- 0.0113 | 1.0000 +/- 0.0000 | 3.0751 +/- 0.1128 | 0.1325 +/- 0.0151 | 0.9078 +/- 0.0192 | 0.0262 +/- 0.0087 |
| 4 | No Regularization + UQ | 0.4408 +/- 0.0115 | 1.0000 +/- 0.0000 | 2.8500 +/- 0.0389 | 0.1446 +/- 0.0198 | 0.7839 +/- 0.0085 | 0.4574 +/- 0.0515 |
| 4 | Gompertz-inspired regularization lambda=1 + UQ | 0.4341 +/- 0.0125 | 1.0000 +/- 0.0000 | 2.8441 +/- 0.1017 | 0.1457 +/- 0.0248 | 0.7770 +/- 0.0223 | 0.0491 +/- 0.0110 |

## Table 4B. Lambda=10 calibrated UQ results

| m | Method | RMSE | PICP | MPIW | ECE | NLL | Gompertz-style residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | No Regularization + UQ | 0.9664 +/- 0.0016 | 1.0000 +/- 0.0000 | 6.3556 +/- 0.0491 | 0.1632 +/- 0.0019 | 1.5799 +/- 0.0055 | 0.0577 +/- 0.0376 |
| 1 | Gompertz-inspired regularization lambda=10 + UQ | 0.9640 +/- 0.0014 | 1.0000 +/- 0.0000 | 6.3389 +/- 0.0273 | 0.1632 +/- 0.0038 | 1.5773 +/- 0.0033 | 0.0037 +/- 0.0014 |
| 2 | No Regularization + UQ | 0.7806 +/- 0.0048 | 0.9825 +/- 0.0152 | 3.5456 +/- 0.2422 | 0.0631 +/- 0.0177 | 1.1885 +/- 0.0169 | 0.3509 +/- 0.0735 |
| 2 | Gompertz-inspired regularization lambda=10 + UQ | 0.7747 +/- 0.0022 | 0.9825 +/- 0.0152 | 3.4788 +/- 0.1084 | 0.0589 +/- 0.0048 | 1.1770 +/- 0.0081 | 0.0051 +/- 0.0011 |
| 3 | No Regularization + UQ | 0.5351 +/- 0.0114 | 1.0000 +/- 0.0000 | 3.2107 +/- 0.0796 | 0.1413 +/- 0.0076 | 0.9285 +/- 0.0201 | 0.4476 +/- 0.0623 |
| 3 | Gompertz-inspired regularization lambda=10 + UQ | 0.5342 +/- 0.0103 | 1.0000 +/- 0.0000 | 3.0947 +/- 0.0944 | 0.1336 +/- 0.0133 | 0.9079 +/- 0.0116 | 0.0132 +/- 0.0048 |
| 4 | No Regularization + UQ | 0.4408 +/- 0.0115 | 1.0000 +/- 0.0000 | 2.8500 +/- 0.0389 | 0.1446 +/- 0.0198 | 0.7839 +/- 0.0085 | 0.4574 +/- 0.0515 |
| 4 | Gompertz-inspired regularization lambda=10 + UQ | 0.4311 +/- 0.0023 | 1.0000 +/- 0.0000 | 2.9453 +/- 0.0516 | 0.1676 +/- 0.0038 | 0.7969 +/- 0.0123 | 0.0214 +/- 0.0059 |

## Table 4C. Lambda sensitivity

| Fixed lambda | RMSE improvement | ECE improvement | NLL improvement | Mean residual ratio | Interpretation |
| --- | --- | --- | --- | --- | --- |
| lambda=1 | 3/4 | 3/4 | 4/4 | 0.062 | Main setting; smaller residual with more stable calibration |
| lambda=10 | 4/4 | 2/4 | 3/4 | 0.039 | Stronger residual reduction, but calibration/loss behavior is less uniform |

## Table 4D. High residual stratum

| m | High-residual RMSE No Regularization | High-residual RMSE regularized | High-residual NLL No Regularization | High-residual NLL regularized | High-residual residual No Regularization | High-residual residual regularized |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8663 | 1.0023 | 1.5457 | 1.5953 | 0.0699 | 0.0057 |
| 2 | 0.8469 | 0.7613 | 1.2534 | 1.1666 | 0.6364 | 0.0139 |
| 3 | 0.6819 | 0.4943 | 1.0746 | 0.8956 | 0.8140 | 0.0424 |
| 4 | 0.4402 | 0.4420 | 0.8039 | 0.7972 | 0.8331 | 0.0879 |

## Table 4E. External trajectory consistency diagnostics

| lambda | m | Method | Mean absolute endpoint jump | Mean absolute slope jump |
| --- | --- | --- | --- | --- |
| 1 | 1 | No Regularization + UQ | 0.0404 +/- 0.0188 | N/A |
| 1 | 1 | Gompertz-inspired regularization lambda=1 + UQ | 0.0224 +/- 0.0144 | N/A |
| 1 | 2 | No Regularization + UQ | 0.1141 +/- 0.0473 | 0.4485 +/- 0.0138 |
| 1 | 2 | Gompertz-inspired regularization lambda=1 + UQ | 0.1026 +/- 0.0323 | 0.4502 +/- 0.0112 |
| 1 | 3 | No Regularization + UQ | 0.0881 +/- 0.0127 | 0.4180 +/- 0.0105 |
| 1 | 3 | Gompertz-inspired regularization lambda=1 + UQ | 0.1299 +/- 0.0422 | 0.4397 +/- 0.0171 |
| 1 | 4 | No Regularization + UQ | 0.0943 +/- 0.0111 | 0.4707 +/- 0.0143 |
| 1 | 4 | Gompertz-inspired regularization lambda=1 + UQ | 0.0761 +/- 0.0110 | 0.4580 +/- 0.0144 |
| 10 | 1 | No Regularization + UQ | 0.0404 +/- 0.0188 | N/A |
| 10 | 1 | Gompertz-inspired regularization lambda=10 + UQ | 0.0227 +/- 0.0135 | N/A |
| 10 | 2 | No Regularization + UQ | 0.1141 +/- 0.0473 | 0.4485 +/- 0.0138 |
| 10 | 2 | Gompertz-inspired regularization lambda=10 + UQ | 0.1003 +/- 0.0307 | 0.4510 +/- 0.0108 |
| 10 | 3 | No Regularization + UQ | 0.0881 +/- 0.0127 | 0.4180 +/- 0.0105 |
| 10 | 3 | Gompertz-inspired regularization lambda=10 + UQ | 0.1150 +/- 0.0386 | 0.4346 +/- 0.0151 |
| 10 | 4 | No Regularization + UQ | 0.0943 +/- 0.0111 | 0.4707 +/- 0.0143 |
| 10 | 4 | Gompertz-inspired regularization lambda=10 + UQ | 0.0596 +/- 0.0065 | 0.4590 +/- 0.0064 |

Figure: `outputs_nlstt_adaptive_uq_paper/experiment4_physics_uq_refined/fig_experiment4_physics_uq_reliability.png`
