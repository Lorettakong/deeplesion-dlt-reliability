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

display_method <- function(x) {
  x <- as.character(x)
  x[x %in% c("Residual Gaussian", "Bayesian/Laplace")] <- "Gaussian residual-scale"
  x
}

method_levels <- c("Deterministic", "Gaussian Process", "MC Dropout", "Deep Ensemble", "Gaussian residual-scale")
method_cols <- c(
  "Deterministic" = "#4C78A8",
  "Gaussian Process" = "#B279A2",
  "MC Dropout" = "#F58518",
  "Deep Ensemble" = "#54A24B",
  "Gaussian residual-scale" = "#E45756"
)
target_cols <- c("90% target" = "#4C78A8", "95% target" = "#F58518")

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

sens_path <- file.path(out_dir, "experiment2_calibration_sensitivity_summary.csv")
sens <- read.csv(sens_path, check.names = FALSE)
sens <- sens[sens$m == 4 & sens$alpha %in% c(0.10, 0.05), ]
sens$method <- display_method(sens$method)
sens$method <- factor(sens$method, levels = method_levels)
sens$target <- ifelse(abs(sens$alpha - 0.10) < 1e-8, "90% target", "95% target")
sens$target <- factor(sens$target, levels = c("90% target", "95% target"))
sens <- sens[order(sens$method, sens$alpha), ]

write.csv(sens, file.path(out_dir, "fig_experiment2_calibration_sensitivity_cmpb_source_data.csv"), row.names = FALSE)

p_a <- ggplot(sens, aes(x = method, y = calibrated_picp_mean, fill = target)) +
  geom_hline(yintercept = 0.90, linetype = "dotted", linewidth = 0.35, colour = "grey45") +
  geom_hline(yintercept = 0.95, linetype = "dashed", linewidth = 0.35, colour = "grey35") +
  geom_col(position = position_dodge(width = 0.72), width = 0.58, colour = "white", linewidth = 0.2) +
  geom_errorbar(
    aes(ymin = picp_wilson95_low_mean, ymax = picp_wilson95_high_mean),
    position = position_dodge(width = 0.72),
    width = 0.14,
    linewidth = 0.35,
    colour = "black"
  ) +
  scale_fill_manual(values = target_cols) +
  scale_y_continuous(breaks = seq(0.75, 1.00, 0.05), expand = expansion(mult = c(0, 0.02))) +
  coord_cartesian(ylim = c(0.75, 1.02)) +
  labs(title = "A. m = 4 calibrated PICP with Wilson CI", x = NULL, y = "PICP") +
  theme_cmpb() +
  theme(axis.text.x = element_text(angle = 22, hjust = 1),
        legend.position = c(0.20, 0.14))

p_b <- ggplot(sens, aes(x = calibrated_mpiw_mean, y = calibrated_picp_mean,
                        colour = method, group = method)) +
  geom_hline(yintercept = 0.95, linetype = "dashed", linewidth = 0.35, colour = "grey35") +
  geom_line(linewidth = 0.55) +
  geom_point(aes(shape = target), size = 2.0) +
  scale_colour_manual(values = method_cols) +
  scale_shape_manual(values = c("90% target" = 16, "95% target" = 17)) +
  scale_y_continuous(limits = c(0.94, 1.01), breaks = seq(0.94, 1.00, 0.02), expand = expansion(mult = c(0.02, 0.03))) +
  scale_x_continuous(expand = expansion(mult = c(0.08, 0.08))) +
  labs(title = "B. m = 4 coverage-width trade-off",
       x = "Mean prediction interval width", y = "PICP") +
  theme_cmpb() +
  guides(
    colour = guide_legend(nrow = 3, byrow = TRUE, override.aes = list(shape = 16, linewidth = 0)),
    shape = guide_legend(nrow = 1)
  ) +
  theme(
    legend.position = c(0.53, 0.23),
    legend.box = "vertical",
    legend.background = element_rect(fill = "white", colour = NA),
    legend.margin = margin(1, 1, 1, 1)
  )

fig <- (p_a | p_b) +
  plot_layout(widths = c(1, 1)) &
  theme(plot.background = element_rect(fill = "white", colour = NA))

pdf_file <- file.path(out_dir, "fig_experiment2_calibration_sensitivity_m4_cmpb.pdf")
png_file <- file.path(out_dir, "fig_experiment2_calibration_sensitivity_m4_cmpb.png")
grDevices::pdf(pdf_file, width = 7.2, height = 3.1, family = "Times", useDingbats = FALSE)
print(fig)
dev.off()
ggsave(png_file, fig, width = 7.2, height = 3.1, dpi = 600, bg = "white")

file.copy(pdf_file, file.path(out_dir, "Figure8.pdf"), overwrite = TRUE)
file.copy(png_file, file.path(out_dir, "Figure8.png"), overwrite = TRUE)
file.copy(pdf_file, file.path(out_dir, "fig_experiment2_calibration_sensitivity_m4.pdf"), overwrite = TRUE)
file.copy(png_file, file.path(out_dir, "fig_experiment2_calibration_sensitivity_m4.png"), overwrite = TRUE)
