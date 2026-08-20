#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(ragg)
})

script_file <- sub("^--file=", "", commandArgs(FALSE)[grep("^--file=", commandArgs(FALSE))][1])
if (is.na(script_file) || !nzchar(script_file)) {
  script_file <- rstudioapi::getActiveDocumentContext()$path
}
root_dir <- normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)
out_dir <- file.path(root_dir, "outputs")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

pdf_path <- file.path(out_dir, "Figure5_experiment0_data_quality_cmpb.pdf")
png_path <- file.path(out_dir, "Figure5_experiment0_data_quality_cmpb.png")
csv_path <- file.path(out_dir, "Figure5_experiment0_data_quality_cmpb_source_data.csv")

# Panel A counts were reconstructed from the original manuscript figure because
# no previous Experiment 0 plotting script was present in the project tree.
trajectory_length_counts <- data.frame(
  unique_ct_scans = factor(
    c("1", "2", "3", "4", "5", "6-8", "9-12", "13-20", ">20"),
    levels = c("1", "2", "3", "4", "5", "6-8", "9-12", "13-20", ">20")
  ),
  components = c(0, 1300, 410, 195, 75, 95, 25, 12, 4)
)

body_site_counts <- data.frame(
  site = factor(c("Chest/lung", "Abdomen/liver", "Other"), levels = c("Chest/lung", "Abdomen/liver", "Other")),
  trajectories = c(90, 70, 45)
)
body_site_counts$percent <- 100 * body_site_counts$trajectories / sum(body_site_counts$trajectories)
body_site_counts$label <- sprintf("%d (%.1f%%)", body_site_counts$trajectories, body_site_counts$percent)

split_counts <- data.frame(
  set = factor(c("Train", "Validation", "Test"), levels = c("Train", "Validation", "Test")),
  Patients = c(84, 19, 26),
  Trajectories = c(140, 27, 38)
)
split_long <- reshape(
  split_counts,
  varying = c("Patients", "Trajectories"),
  v.names = "count",
  timevar = "count_type",
  times = c("Patients", "Trajectories"),
  direction = "long"
)
split_long$count_type <- factor(split_long$count_type, levels = c("Patients", "Trajectories"))

outlier_rates <- data.frame(
  variable = factor(
    c("All V(t)", "All log V(t)", "All relative log V", "T4 V(t)", "T4 log V(t)", "T4 relative log V"),
    levels = c("All V(t)", "All log V(t)", "All relative log V", "T4 V(t)", "T4 log V(t)", "T4 relative log V")
  ),
  iqr_outlier_rate = c(15.0, 0.9, 8.4, 13.7, 1.0, 2.4),
  family = factor(
    c("Volume", "Log-volume", "Relative log-volume", "Volume", "Log-volume", "Relative log-volume"),
    levels = c("Volume", "Log-volume", "Relative log-volume")
  )
)

source_data <- rbind(
  data.frame(panel = "A", metric = "candidate trajectory length", item = as.character(trajectory_length_counts$unique_ct_scans), value = trajectory_length_counts$components),
  data.frame(panel = "B", metric = "body-site distribution", item = as.character(body_site_counts$site), value = body_site_counts$trajectories),
  data.frame(panel = "C", metric = "patient-level split - patients", item = as.character(split_counts$set), value = split_counts$Patients),
  data.frame(panel = "C", metric = "patient-level split - trajectories", item = as.character(split_counts$set), value = split_counts$Trajectories),
  data.frame(panel = "D", metric = "IQR outlier rate (%)", item = as.character(outlier_rates$variable), value = outlier_rates$iqr_outlier_rate)
)
write.csv(source_data, csv_path, row.names = FALSE)

palette_main <- c(
  blue = "#4C78A8",
  orange = "#F58518",
  green = "#54A24B",
  purple = "#9D77C9",
  gray = "#8A8A8A"
)
blue <- unname(palette_main["blue"])
orange <- unname(palette_main["orange"])
green <- unname(palette_main["green"])
purple <- unname(palette_main["purple"])

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
      legend.position = "top",
      legend.title = element_blank(),
      legend.key.size = unit(0.35, "lines"),
      legend.text = element_text(family = "Times", size = 7),
      plot.margin = margin(4, 5, 4, 5)
    )
}

panel_a <- ggplot(trajectory_length_counts, aes(x = unique_ct_scans, y = components)) +
  geom_col(width = 0.72, fill = blue, colour = "white", linewidth = 0.15) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.04)), limits = c(0, 1420)) +
  labs(
    title = "A. Candidate trajectory length",
    x = "Unique CT scans per connected component",
    y = "Connected components"
  ) +
  theme_cmpb()

panel_b <- ggplot(body_site_counts, aes(x = site, y = trajectories, fill = site)) +
  geom_col(width = 0.58, colour = "white", linewidth = 0.15) +
  geom_text(aes(label = label), vjust = -0.35, size = 2.35) +
  scale_fill_manual(values = c("Chest/lung" = blue, "Abdomen/liver" = green, "Other" = purple)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08)), limits = c(0, 105)) +
  guides(fill = "none") +
  labs(title = "B. Body-site distribution", x = NULL, y = "Trajectories") +
  theme_cmpb()

panel_c <- ggplot(split_long, aes(x = set, y = count, fill = count_type)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.32, colour = "white", linewidth = 0.15) +
  scale_fill_manual(values = c("Patients" = blue, "Trajectories" = orange)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.06)), limits = c(0, 155)) +
  labs(title = "C. Patient-level split", x = NULL, y = "Count") +
  theme_cmpb()

panel_d <- ggplot(outlier_rates, aes(x = variable, y = iqr_outlier_rate, fill = family)) +
  geom_col(width = 0.66, colour = "white", linewidth = 0.15) +
  geom_text(aes(label = sprintf("%.1f", iqr_outlier_rate)), vjust = -0.35, size = 2.2) +
  scale_fill_manual(values = c("Volume" = purple, "Log-volume" = blue, "Relative log-volume" = green)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08)), limits = c(0, 16.6)) +
  labs(title = "D. Outlier audit", x = NULL, y = "IQR outlier rate (%)") +
  theme_cmpb() +
  theme(
    axis.text.x = element_text(angle = 28, hjust = 1, vjust = 1),
    legend.position = "top"
  )

figure <- (panel_a | panel_b) / (panel_c | panel_d) +
  plot_layout(guides = "keep") &
  theme(plot.background = element_rect(fill = "white", colour = NA))

ggsave(
  pdf_path,
  figure,
  width = 7.2,
  height = 5.25,
  units = "in",
  device = function(filename, width, height, ...) {
    grDevices::pdf(filename, width = width, height = height, family = "Times", useDingbats = FALSE, ...)
  }
)
ggsave(png_path, figure, width = 7.2, height = 5.25, units = "in", dpi = 600, device = ragg::agg_png)

message("Wrote ", pdf_path)
message("Wrote ", png_path)
message("Wrote ", csv_path)
