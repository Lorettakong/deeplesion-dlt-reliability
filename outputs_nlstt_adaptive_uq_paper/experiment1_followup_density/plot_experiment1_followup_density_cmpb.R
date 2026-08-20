#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(ragg)
})

script_file <- sub("^--file=", "", commandArgs(FALSE)[grep("^--file=", commandArgs(FALSE))][1])
script_file <- gsub("~\\+~", " ", script_file, fixed = FALSE)
if (is.na(script_file) || !nzchar(script_file)) {
  stop("Run this script with Rscript so the output directory can be resolved.")
}

out_dir <- normalizePath(dirname(script_file), mustWork = TRUE)
root_dir <- normalizePath(file.path(out_dir, ".."), mustWork = TRUE)
bootstrap_dir <- file.path(root_dir, "patient_cluster_bootstrap")

pooled_path <- file.path(bootstrap_dir, "experiment1_neural_patient_cluster_bootstrap.csv")
subgroup_path <- file.path(bootstrap_dir, "experiment1_subgroup_patient_cluster_bootstrap.csv")

pooled <- read.csv(pooled_path, check.names = FALSE)
subgroup <- read.csv(subgroup_path, check.names = FALSE)

pooled <- pooled[pooled$analysis == "pooled test", ]
pooled$m <- as.numeric(pooled$m)
pooled$label <- sprintf("%.3f", pooled$rmse_mean)

subgroup <- subgroup[subgroup$analysis == "subgroup exploratory", ]
subgroup$m <- as.numeric(subgroup$m)
subgroup$subgroup <- factor(subgroup$subgroup, levels = c("chest/lung", "abdomen/liver", "other"))
subgroup$legend_label <- paste0(subgroup$subgroup, " (N=", subgroup$N_test_trajectories, ")")
subgroup$legend_label <- factor(
  subgroup$legend_label,
  levels = c("chest/lung (N=17)", "abdomen/liver (N=13)", "other (N=8)")
)

source_data <- rbind(
  data.frame(
    panel = "A",
    group = "pooled test",
    m = pooled$m,
    rmse = pooled$rmse_mean,
    ci95_low = pooled$rmse_ci95_low,
    ci95_high = pooled$rmse_ci95_high
  ),
  data.frame(
    panel = "B",
    group = as.character(subgroup$subgroup),
    m = subgroup$m,
    rmse = subgroup$rmse_mean,
    ci95_low = subgroup$rmse_ci95_low,
    ci95_high = subgroup$rmse_ci95_high
  )
)
write.csv(source_data, file.path(out_dir, "fig_experiment1_followup_density_cmpb_source_data.csv"), row.names = FALSE)

blue <- "#4C78A8"
green <- "#54A24B"
purple <- "#9D77C9"
gray <- "#6B6B6B"

theme_cmpb <- function(base_size = 8) {
  theme_classic(base_size = base_size, base_family = "Times") +
    theme(
      text = element_text(family = "Times"),
      plot.title = element_text(family = "Times", face = "bold", size = 9, hjust = 0, margin = margin(b = 4)),
      axis.title = element_text(family = "Times", size = 8),
      axis.text = element_text(family = "Times", size = 7, colour = "black"),
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.3, colour = "black"),
      panel.grid.major.y = element_line(linewidth = 0.25, colour = "#D9D9D9"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      legend.title = element_blank(),
      legend.text = element_text(family = "Times", size = 7),
      legend.key.height = unit(0.35, "lines"),
      legend.key.width = unit(0.7, "lines"),
      plot.margin = margin(4, 5, 4, 5)
    )
}

panel_a <- ggplot(pooled, aes(x = m, y = rmse_mean)) +
  geom_errorbar(aes(ymin = rmse_ci95_low, ymax = rmse_ci95_high), width = 0.08, linewidth = 0.35, colour = gray) +
  geom_line(linewidth = 0.65, colour = blue) +
  geom_point(size = 1.9, colour = blue) +
  geom_text(aes(label = label), vjust = -1.05, size = 2.35, family = "Times") +
  scale_x_continuous(breaks = 1:4, limits = c(0.85, 4.15), expand = expansion(mult = c(0.02, 0.02))) +
  scale_y_continuous(limits = c(0.30, 1.12), expand = expansion(mult = c(0, 0.03))) +
  labs(
    title = "A. Pooled validation-selected RMSE",
    x = "Observed CT visits (m)",
    y = "RMSE of relative log-volume"
  ) +
  theme_cmpb()

panel_b <- ggplot(subgroup, aes(x = m, y = rmse_mean, colour = legend_label, shape = legend_label)) +
  geom_errorbar(aes(ymin = rmse_ci95_low, ymax = rmse_ci95_high), width = 0.08, linewidth = 0.3, alpha = 0.8) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.8) +
  scale_colour_manual(
    values = c(
      "chest/lung (N=17)" = blue,
      "abdomen/liver (N=13)" = green,
      "other (N=8)" = purple
    )
  ) +
  scale_shape_manual(values = c("chest/lung (N=17)" = 16, "abdomen/liver (N=13)" = 17, "other (N=8)" = 15)) +
  scale_x_continuous(breaks = 1:4, limits = c(0.85, 4.15), expand = expansion(mult = c(0.02, 0.02))) +
  scale_y_continuous(limits = c(0.15, 1.35), expand = expansion(mult = c(0, 0.03))) +
  labs(
    title = "B. Subgroup RMSE by lesion site",
    x = "Observed CT visits (m)",
    y = "RMSE of relative log-volume"
  ) +
  guides(
    colour = guide_legend(nrow = 1, byrow = TRUE),
    shape = guide_legend(nrow = 1, byrow = TRUE)
  ) +
  theme_cmpb() +
  theme(
    legend.position = c(0.52, 0.97),
    legend.justification = c(0.5, 1),
    legend.background = element_rect(fill = "white", colour = NA)
  )

figure <- (panel_a | panel_b) +
  plot_layout(widths = c(1, 1)) &
  theme(plot.background = element_rect(fill = "white", colour = NA))

pdf_path <- file.path(out_dir, "fig_experiment1_followup_density_cmpb.pdf")
png_path <- file.path(out_dir, "fig_experiment1_followup_density_cmpb.png")

ggsave(
  pdf_path,
  figure,
  width = 7.2,
  height = 3.0,
  units = "in",
  device = function(filename, width, height, ...) {
    grDevices::pdf(filename, width = width, height = height, family = "Times", useDingbats = FALSE, ...)
  }
)
ggsave(png_path, figure, width = 7.2, height = 3.0, units = "in", dpi = 600, device = ragg::agg_png)

file.copy(pdf_path, file.path(out_dir, "Figure6.pdf"), overwrite = TRUE)
file.copy(png_path, file.path(out_dir, "Figure6.png"), overwrite = TRUE)

message("Wrote ", pdf_path)
message("Wrote ", png_path)
message("Wrote ", file.path(out_dir, "Figure6.pdf"))
message("Wrote ", file.path(out_dir, "Figure6.png"))
