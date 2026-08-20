#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_r() {
  local script_path="$1"
  echo "Running ${script_path}"
  Rscript "${ROOT}/${script_path}"
}

run_r "outputs_nlstt_adaptive_uq_paper/experiment0_data_quality_audit/plot_experiment0_data_quality_cmpb.R"
run_r "outputs_nlstt_adaptive_uq_paper/experiment1_followup_density/plot_experiment1_followup_density_cmpb.R"
run_r "outputs_nlstt_adaptive_uq_paper/experiment2_uq_refined/plot_experiment2_uq_refined_cmpb.R"
run_r "outputs_nlstt_adaptive_uq_paper/experiment2_calibration_sensitivity/plot_experiment2_calibration_sensitivity_cmpb.R"
run_r "outputs_nlstt_adaptive_uq_paper/experiment3_physics_refined/plot_experiment3_physics_tradeoff_cmpb.R"
run_r "outputs_nlstt_adaptive_uq_paper/experiment4_physics_uq_refined/plot_experiment4_uq_reliability_cmpb.R"
run_r "outputs_nlstt_adaptive_uq_paper/experiment5_calibration_diagnostics/plot_experiment5_calibration_diagnostics_cmpb.R"

echo "Done. Polished figure PDFs/PNGs were regenerated under outputs_nlstt_adaptive_uq_paper/."
