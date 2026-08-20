script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", script_args, value = TRUE)
if (length(file_arg)) {
  script_path <- normalizePath(gsub("~\\+~", " ", sub("^--file=", "", file_arg[1])), mustWork = TRUE)
  out_dir <- dirname(script_path)
} else {
  out_dir <- getwd()
}

root_dir <- normalizePath(file.path(out_dir, ".."), mustWork = TRUE)
pred_dir <- file.path(root_dir, "experiment2_uq_refined")

method_ids <- c("deterministic", "mc_dropout", "deep_ensemble", "bayesian_laplace")
method_labels <- c("Deterministic", "MC Dropout", "Deep Ensemble", "Residual Gaussian")
names(method_labels) <- method_ids

palette <- c(
  "Deterministic" = "#2F6FB0",
  "MC Dropout" = "#E9842A",
  "Deep Ensemble" = "#4C9A3F",
  "Residual Gaussian" = "#C83F49"
)

map_method <- function(x) {
  x <- gsub("Residual Gaussian", "Residual Gaussian", x, fixed = TRUE)
  x
}

read_predictions <- function() {
  all <- list()
  for (rep in paste0("repeat", 1:3)) {
    m_dir <- file.path(pred_dir, rep, "m4")
    for (mid in method_ids) {
      f <- file.path(m_dir, paste0("pred_", mid, "_calibrated.csv"))
      if (!file.exists(f)) next
      raw <- read.csv(f, check.names = FALSE)
      d <- data.frame(
        run_id = rep,
        trajectory_id = raw$trajectory_id,
        method = method_labels[[mid]],
        logv_target = raw$logv_target,
        logv_mean = raw$logv_mean,
        logv_lower = raw$logv_lower,
        logv_upper = raw$logv_upper,
        stringsAsFactors = FALSE
      )
      d$abs_error <- abs(d$logv_target - d$logv_mean)
      d$interval_width <- d$logv_upper - d$logv_lower
      d$half_width <- d$interval_width / 2
      all[[length(all) + 1]] <- d
    }
  }
  do.call(rbind, all)
}

parse_ci_low <- function(x) {
  as.numeric(sub("-.*$", "", x))
}

pred <- read_predictions()
pred$method <- factor(pred$method, levels = method_labels)

subgroup <- read.csv(file.path(out_dir, "table5b_subgroup_calibration.csv"), check.names = FALSE)
subgroup$Method <- map_method(subgroup$Method)
subgroup$Method <- factor(subgroup$Method, levels = method_labels)
subgroup$Subgroup <- factor(subgroup$Subgroup, levels = c("chest/lung", "abdomen/liver", "other"))
subgroup$ci_low <- parse_ci_low(subgroup[["PICP Clopper-Pearson 95% CI"]])

target <- read.csv(file.path(out_dir, "table5c_target_magnitude_coverage.csv"), check.names = FALSE)
target$Method <- map_method(target$Method)
target$Method <- factor(target$Method, levels = method_labels)
target[["Target magnitude"]] <- factor(
  target[["Target magnitude"]],
  levels = c("low y(T4)", "middle y(T4)", "high y(T4)"),
  labels = c("low y(T4)", "middle y(T4)", "high y(T4)")
)
target$ci_low <- parse_ci_low(target[["PICP Clopper-Pearson 95% CI"]])

reliability_levels <- c(0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
z95 <- qnorm(0.975)
rel_rows <- list()
for (m in levels(pred$method)) {
  d <- pred[pred$method == m, ]
  for (nom in reliability_levels) {
    z_nom <- qnorm((1 + nom) / 2)
    covered <- mean(d$abs_error <= d$half_width * z_nom / z95)
    rel_rows[[length(rel_rows) + 1]] <- data.frame(method = m, nominal = nom, empirical = covered)
  }
}
reliability <- do.call(rbind, rel_rows)

write.csv(pred[, c("run_id", "trajectory_id", "method", "logv_target", "logv_mean",
                  "logv_lower", "logv_upper", "abs_error", "interval_width")],
          file.path(out_dir, "figure11_cmpb_prediction_source_data.csv"), row.names = FALSE)
write.csv(reliability, file.path(out_dir, "figure11_cmpb_reliability_source_data.csv"), row.names = FALSE)

draw_panel_label <- function(label) {
  title(label, adj = 0, line = 0.45, cex.main = 1.05, font.main = 2)
}

thin_axes <- function() {
  box(lwd = 0.65)
}

plot_reliability <- function() {
  par(mar = c(3.7, 4.0, 2.2, 0.8))
  plot(NA, xlim = c(0.48, 0.97), ylim = c(0.48, 1.02), xaxt = "n", yaxt = "n",
       xlab = "Nominal coverage", ylab = "Empirical coverage", bty = "n")
  grid(col = "#E6E6E6", lwd = 0.55)
  abline(0, 1, lty = 2, lwd = 0.7, col = "#555555")
  for (m in levels(pred$method)) {
    d <- reliability[reliability$method == m, ]
    lines(d$nominal, d$empirical, col = palette[[m]], lwd = 1.4)
    points(d$nominal, d$empirical, col = palette[[m]], pch = 16, cex = 0.7)
  }
  thin_axes()
  axis(1, at = reliability_levels, labels = sprintf("%.2g", reliability_levels),
       lwd = 0.65, lwd.ticks = 0.65)
  axis(2, at = seq(0.5, 1.0, 0.1), las = 1, lwd = 0.65, lwd.ticks = 0.65)
  legend("bottomright", legend = levels(pred$method), col = palette[levels(pred$method)],
         lty = 1, pch = 16, bty = "n", cex = 0.72, lwd = 1.2, y.intersp = 0.88)
  draw_panel_label("A. Reliability diagram")
}

plot_scatter <- function() {
  par(mar = c(3.7, 4.0, 2.2, 0.8))
  plot(NA, xlim = range(pred$interval_width) + c(-0.05, 0.15),
       ylim = c(0, max(pred$abs_error) * 1.08), xaxt = "n", yaxt = "n",
       xlab = "Calibrated interval width", ylab = "Absolute prediction error", bty = "n")
  grid(col = "#E6E6E6", lwd = 0.55)
  for (m in levels(pred$method)) {
    d <- pred[pred$method == m, ]
    points(d$interval_width, d$abs_error, pch = 16, cex = 0.55,
           col = adjustcolor(palette[[m]], alpha.f = 0.62))
  }
  thin_axes()
  axis(1, lwd = 0.65, lwd.ticks = 0.65)
  axis(2, las = 1, lwd = 0.65, lwd.ticks = 0.65)
  draw_panel_label("B. Interval width vs. error")
}

plot_box <- function() {
  par(mar = c(3.7, 4.0, 2.2, 0.8))
  vals <- split(pred$interval_width, pred$method)
  axis_labels <- c("Det.", "MC Dropout", "Ensemble", "Residual G.")
  boxplot(vals, border = "#555555", col = adjustcolor(palette[names(vals)], alpha.f = 0.18),
          outline = FALSE, xaxt = "n", yaxt = "n", ylab = "Calibrated interval width",
          frame.plot = FALSE, lwd = 0.75, whisklty = 1, staplewex = 0.5, medlwd = 1.0)
  grid(nx = NA, ny = NULL, col = "#E6E6E6", lwd = 0.55)
  axis(1, at = seq_along(vals), labels = axis_labels, las = 1, cex.axis = 0.78,
       lwd = 0.65, lwd.ticks = 0.65)
  axis(2, las = 1, lwd = 0.65, lwd.ticks = 0.65)
  box(lwd = 0.65)
  draw_panel_label("C. Interval width distribution")
}

plot_group_ci <- function(df, group_col, panel_title, axis_labels = NULL, xlab_cex = 0.68) {
  groups <- levels(df[[group_col]])
  if (is.null(axis_labels)) axis_labels <- groups
  methods <- levels(df$Method)
  par(mar = c(3.7, 4.0, 2.2, 0.8))
  plot(NA, xlim = c(0.5, length(groups) + 0.5), ylim = c(0.56, 1.03), xaxt = "n", yaxt = "n",
       xlab = "", ylab = "PICP (exact 95% CI)", bty = "n")
  grid(nx = NA, ny = NULL, col = "#E6E6E6", lwd = 0.55)
  abline(h = 0.95, lty = 2, col = "#666666", lwd = 0.8)
  offsets <- seq(-0.24, 0.24, length.out = length(methods))
  for (i in seq_along(groups)) {
    for (j in seq_along(methods)) {
      row <- df[df[[group_col]] == groups[i] & df$Method == methods[j], ]
      x <- i + offsets[j]
      segments(x, row$ci_low, x, row$PICP, col = palette[[methods[j]]], lwd = 1.0)
      segments(x - 0.035, row$ci_low, x + 0.035, row$ci_low, col = palette[[methods[j]]], lwd = 1.0)
      points(x, row$PICP, pch = 16, col = palette[[methods[j]]], cex = 0.74)
    }
  }
  axis(1, at = seq_along(groups), labels = axis_labels, las = 1, cex.axis = xlab_cex,
       lwd = 0.65, lwd.ticks = 0.65)
  axis(2, at = seq(0.6, 1.0, 0.1), las = 1, lwd = 0.65, lwd.ticks = 0.65)
  box(lwd = 0.65)
  draw_panel_label(panel_title)
}

pdf_file <- file.path(out_dir, "Figure11.pdf")
pdf(pdf_file, width = 7.2, height = 5.6, family = "Times", pointsize = 9.5, useDingbats = FALSE)
op <- par(no.readonly = TRUE)
layout(matrix(c(1, 1, 2, 2, 3, 3,
                4, 4, 4, 5, 5, 5), nrow = 2, byrow = TRUE),
       widths = rep(1, 6), heights = c(1, 1.08))
par(family = "Times", oma = c(0.25, 0.2, 0.25, 0.2), mgp = c(2.25, 0.65, 0), tcl = -0.25,
    cex.axis = 0.88, cex.lab = 0.96)
plot_reliability()
plot_scatter()
plot_box()
plot_group_ci(subgroup, "Subgroup", "D. Subgroup PICP",
              axis_labels = c("Chest/lung", "Abd./liver", "Other"), xlab_cex = 0.82)
plot_group_ci(target, "Target magnitude", "E. Target-magnitude PICP",
              axis_labels = c("Low", "Middle", "High"), xlab_cex = 0.82)
par(op)
dev.off()

file.copy(pdf_file, file.path(out_dir, "fig_experiment5_calibration_diagnostics_cmpb.pdf"), overwrite = TRUE)
file.copy(pdf_file, file.path(out_dir, "fig_experiment5_calibration_diagnostics.pdf"), overwrite = TRUE)

pdftoppm <- Sys.which("pdftoppm")
if (nzchar(pdftoppm)) {
  prefix <- file.path(out_dir, "Figure11")
  system2(pdftoppm, c("-png", "-r", "600", "-singlefile", shQuote(pdf_file), shQuote(prefix)))
  file.copy(file.path(out_dir, "Figure11.png"),
            file.path(out_dir, "fig_experiment5_calibration_diagnostics_cmpb.png"), overwrite = TRUE)
  file.copy(file.path(out_dir, "Figure11.png"),
            file.path(out_dir, "fig_experiment5_calibration_diagnostics.png"), overwrite = TRUE)
}

cat("Wrote:", pdf_file, "\n")
