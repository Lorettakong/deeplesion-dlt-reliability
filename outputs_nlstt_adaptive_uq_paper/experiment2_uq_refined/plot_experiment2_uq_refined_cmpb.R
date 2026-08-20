library(ggplot2)
library(patchwork)

script_path <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA)
if (is.na(script_path)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script_path <- if (length(file_arg)) {
    normalizePath(gsub("~\\+~", " ", sub("^--file=", "", file_arg[1])), mustWork = TRUE)
  } else {
    getwd()
  }
}
out_dir <- dirname(script_path)
base_dir <- dirname(out_dir)

parse_interval <- function(x) {
  x <- as.character(x)
  is_missing <- is.na(x) | x == "" | x == "NaN"
  mean <- low <- high <- rep(NA_real_, length(x))

  has_ci <- grepl("\\(", x) & grepl("-", x)
  if (any(has_ci & !is_missing)) {
    mean[has_ci] <- as.numeric(sub("^\\s*([0-9.]+).*", "\\1", x[has_ci]))
    inside <- sub(".*\\(([^()]*)\\).*", "\\1", x[has_ci])
    low[has_ci] <- as.numeric(sub("^\\s*([0-9.]+)\\s*-.*", "\\1", inside))
    high[has_ci] <- as.numeric(sub(".*-\\s*([0-9.]+)\\s*$", "\\1", inside))
  }

  has_pm <- grepl("\\+/-", x)
  if (any(has_pm & !is_missing)) {
    mean[has_pm] <- as.numeric(sub("^\\s*([0-9.]+).*", "\\1", x[has_pm]))
    half <- as.numeric(sub(".*\\+/-\\s*([0-9.]+)\\s*$", "\\1", x[has_pm]))
    low[has_pm] <- mean[has_pm] - half
    high[has_pm] <- mean[has_pm] + half
  }

  data.frame(mean = mean, low = low, high = high)
}

display_method <- function(x) {
  x <- as.character(x)
  x[x == "Residual Gaussian"] <- "Residual Gaussian"
  x
}

method_levels <- c("Deterministic", "MC Dropout", "Deep Ensemble", "Residual Gaussian")
method_cols <- c(
  "Deterministic" = "#4C78A8",
  "MC Dropout" = "#F58518",
  "Deep Ensemble" = "#54A24B",
  "Residual Gaussian" = "#E45756"
)
variant_cols <- c("Raw" = "#4C78A8", "Calibrated" = "#F58518")

theme_cmpb <- function(base_size = 8.5) {
  theme_classic(base_size = base_size, base_family = "Times") +
    theme(
      text = element_text(family = "Times", colour = "black"),
      plot.title = element_text(face = "bold", size = base_size + 0.5, hjust = 0),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 0.5, colour = "black"),
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 0.8),
      legend.key.size = unit(0.35, "lines"),
      legend.background = element_blank(),
      panel.grid.major = element_line(colour = "grey88", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      axis.line = element_line(linewidth = 0.35),
      axis.ticks = element_line(linewidth = 0.35),
      plot.margin = margin(4, 5, 4, 4)
    )
}

boot_path <- file.path(base_dir, "patient_cluster_bootstrap", "table2a_calibrated_rmse_patient_cluster_bootstrap.csv")
rmse_wide <- read.csv(boot_path, check.names = FALSE)
rmse_long <- do.call(rbind, lapply(setdiff(names(rmse_wide), "m"), function(method) {
  parsed <- parse_interval(rmse_wide[[method]])
  data.frame(
    m = rmse_wide$m,
    method = display_method(method),
    mean = parsed$mean,
    low = parsed$low,
    high = parsed$high
  )
}))
rmse_long$method <- factor(rmse_long$method, levels = method_levels)

uq_path <- file.path(out_dir, "table2b_m4_raw_vs_calibrated_uq.csv")
uq <- read.csv(uq_path, check.names = FALSE)
uq$method <- display_method(uq$method)
uq$variant_label <- ifelse(uq$variant == "raw", "Raw", "Calibrated")
for (metric in c("PICP", "MPIW", "ECE", "NLL")) {
  parsed <- parse_interval(uq[[metric]])
  uq[[paste0(tolower(metric), "_mean")]] <- parsed$mean
  uq[[paste0(tolower(metric), "_low")]] <- parsed$low
  uq[[paste0(tolower(metric), "_high")]] <- parsed$high
}
uq$method <- factor(uq$method, levels = method_levels)
uq$variant_label <- factor(uq$variant_label, levels = c("Raw", "Calibrated"))

sens_path <- file.path(out_dir, "mc_dropout_sensitivity_summary.csv")
sens <- read.csv(sens_path, check.names = FALSE)
sens$dropout_label <- factor(paste0("p = ", sprintf("%.2f", sens$dropout)),
                             levels = paste0("p = ", sprintf("%.2f", sort(unique(sens$dropout)))))

p_a <- ggplot(rmse_long, aes(x = m, y = mean, colour = method, group = method)) +
  geom_ribbon(aes(ymin = low, ymax = high, fill = method), alpha = 0.08, colour = NA, show.legend = FALSE) +
  geom_errorbar(aes(ymin = low, ymax = high), width = 0.06, linewidth = 0.35, alpha = 0.85) +
  geom_line(linewidth = 0.55) +
  geom_point(size = 1.8) +
  scale_colour_manual(values = method_cols) +
  scale_fill_manual(values = method_cols) +
  scale_x_continuous(breaks = 1:4) +
  coord_cartesian(ylim = c(0.34, 1.12)) +
  labs(title = "A. Test RMSE", x = "Observed CT visits (m)", y = "RMSE") +
  theme_cmpb() +
  theme(legend.position = c(0.75, 0.77), legend.justification = c(0, 0.5))

picp_data <- uq[!is.na(uq$picp_mean), ]
p_b <- ggplot(picp_data, aes(x = method, y = picp_mean, fill = variant_label)) +
  geom_hline(yintercept = 0.95, linetype = "dashed", linewidth = 0.35, colour = "grey35") +
  geom_col(position = position_dodge(width = 0.72), width = 0.58, colour = "white", linewidth = 0.2) +
  scale_fill_manual(values = variant_cols) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.25), expand = expansion(mult = c(0, 0.02))) +
  labs(title = "B. m = 4 PICP", x = NULL, y = "PICP") +
  theme_cmpb() +
  theme(axis.text.x = element_text(angle = 22, hjust = 1),
        legend.position = c(0.73, 0.16))

mpiw_data <- uq[!is.na(uq$mpiw_mean), ]
p_c <- ggplot(mpiw_data, aes(x = method, y = mpiw_mean, fill = variant_label)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.58, colour = "white", linewidth = 0.2) +
  scale_fill_manual(values = variant_cols) +
  scale_y_continuous(limits = c(0, 3.45), breaks = seq(0, 3, 0.75), expand = expansion(mult = c(0, 0.03))) +
  labs(title = "C. m = 4 MPIW", x = NULL, y = "Mean prediction interval width") +
  theme_cmpb() +
  theme(axis.text.x = element_text(angle = 22, hjust = 1),
        legend.position = c(0.73, 0.83))

p_d <- ggplot(sens, aes(x = samples, y = picp, colour = dropout_label, group = dropout_label)) +
  geom_hline(yintercept = 0.95, linetype = "dashed", linewidth = 0.35, colour = "grey35") +
  geom_line(linewidth = 0.55) +
  geom_point(size = 1.65) +
  scale_colour_manual(values = c("#4C78A8", "#F58518", "#54A24B", "#E45756")) +
  scale_x_continuous(breaks = c(30, 50, 100)) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.25), expand = expansion(mult = c(0, 0.02))) +
  labs(title = "D. Raw MC Dropout sensitivity, m = 4",
       x = "Stochastic forward passes", y = "Raw PICP") +
  theme_cmpb() +
  theme(legend.position = c(0.72, 0.27))

fig <- (p_a | p_b) / (p_c | p_d) +
  plot_layout(guides = "keep") &
  theme(plot.background = element_rect(fill = "white", colour = NA))

write.csv(rmse_long, file.path(out_dir, "fig_experiment2_uq_refined_cmpb_rmse_source.csv"), row.names = FALSE)
write.csv(uq, file.path(out_dir, "fig_experiment2_uq_refined_cmpb_uq_source.csv"), row.names = FALSE)
write.csv(sens, file.path(out_dir, "fig_experiment2_uq_refined_cmpb_mc_dropout_source.csv"), row.names = FALSE)

pdf_file <- file.path(out_dir, "fig_experiment2_uq_refined_cmpb.pdf")
png_file <- file.path(out_dir, "fig_experiment2_uq_refined_cmpb.png")
grDevices::pdf(pdf_file, width = 7.2, height = 5.15, family = "Times", useDingbats = FALSE)
print(fig)
dev.off()
ggsave(png_file, fig, width = 7.2, height = 5.15, dpi = 600, bg = "white")

file.copy(pdf_file, file.path(out_dir, "Figure7.pdf"), overwrite = TRUE)
file.copy(png_file, file.path(out_dir, "Figure7.png"), overwrite = TRUE)
file.copy(pdf_file, file.path(out_dir, "fig_experiment2_uq_refined.pdf"), overwrite = TRUE)
file.copy(png_file, file.path(out_dir, "fig_experiment2_uq_refined.png"), overwrite = TRUE)
