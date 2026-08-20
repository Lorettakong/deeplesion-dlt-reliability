# Figures

This directory points to the manuscript figure source data and polished figure
generation scripts.

| Figure | Source folder |
|---|---|
| Figure 5: data quality audit | `outputs_nlstt_adaptive_uq_paper/experiment0_data_quality_audit/` |
| Figure 6: follow-up density | `outputs_nlstt_adaptive_uq_paper/experiment1_followup_density/` |
| Figure 7: UQ comparison | `outputs_nlstt_adaptive_uq_paper/experiment2_uq_refined/` |
| Figure 8: calibration sensitivity | `outputs_nlstt_adaptive_uq_paper/experiment2_calibration_sensitivity/` |
| Figure 9: regularization-weight sensitivity | `outputs_nlstt_adaptive_uq_paper/experiment3_physics_refined/` |
| Figure 10: regularized UQ reliability | `outputs_nlstt_adaptive_uq_paper/experiment4_physics_uq_refined/` |
| Figure 11: calibration/subgroup diagnostics | `outputs_nlstt_adaptive_uq_paper/experiment5_calibration_diagnostics/` |

Each source folder contains the polished PDF/PNG, source CSV files, and the R
script used to regenerate the figure. The convenience command is:

```bash
bash reproduce_figures.sh
```
