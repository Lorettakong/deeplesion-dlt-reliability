script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", script_args, value = TRUE)
if (length(file_arg)) {
  script_path <- normalizePath(gsub("~\\+~", " ", sub("^--file=", "", file_arg[1])), mustWork = TRUE)
  out_dir <- dirname(script_path)
} else {
  out_dir <- getwd()
}

lambda1_path <- file.path(out_dir, "table4a_lambda1_calibrated_uq.csv")
lambda10_path <- file.path(out_dir, "table4b_lambda10_calibrated_uq.csv")

parse_pm <- function(x) {
  x <- trimws(as.character(x))
  parts <- strsplit(x, "\\s*\\+/-\\s*")
  mean <- as.numeric(vapply(parts, `[`, character(1), 1))
  half <- as.numeric(vapply(parts, function(z) ifelse(length(z) >= 2, z[2], NA_character_), character(1)))
  data.frame(mean = mean, half = half)
}

make_long <- function(path, lambda_label) {
  raw <- read.csv(path, check.names = FALSE)
  metrics <- c("RMSE", "Interval score", "WIS (one level)", "Gompertz-style residual")
  rows <- list()
  for (metric in metrics) {
    parsed <- parse_pm(raw[[metric]])
    method_group <- ifelse(
      grepl("^(No Regularization|No Physics)", raw$Method),
      "Unregularized MC Dropout (lambda = 0)",
      lambda_label
    )
    rows[[metric]] <- data.frame(
      m = raw$m,
      method_group = method_group,
      metric = metric,
      mean = parsed$mean,
      half = parsed$half,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

d1 <- make_long(lambda1_path, "lambda = 1")
d10 <- make_long(lambda10_path, "lambda = 10")
plot_data <- rbind(
  d1[d1$method_group %in% c("Unregularized MC Dropout (lambda = 0)", "lambda = 1"), ],
  d10[d10$method_group == "lambda = 10", ]
)
plot_data$method_group <- factor(
  plot_data$method_group,
  levels = c("Unregularized MC Dropout (lambda = 0)", "lambda = 1", "lambda = 10")
)

source_out <- file.path(out_dir, "figure10_cmpb_source_data.csv")
write.csv(plot_data, source_out, row.names = FALSE)

palette <- c(
  "Unregularized MC Dropout (lambda = 0)" = "#1F77B4",
  "lambda = 1" = "#FF7F0E",
  "lambda = 10" = "#2CA02C"
)
shape_map <- c("Unregularized MC Dropout (lambda = 0)" = 16, "lambda = 1" = 17, "lambda = 10" = 15)

draw_errorbar <- function(x, lo, hi, col, log_y = FALSE, width = 0.045) {
  if (log_y) {
    lo <- pmax(lo, 1e-5)
    hi <- pmax(hi, 1e-5)
  }
  arrows(x, lo, x, hi, angle = 90, code = 3, length = width, col = col, lwd = 0.7)
}

draw_metric_panel <- function(metric, title, ylab, ylim = NULL, log_y = FALSE, legend_pos = NULL) {
  d_metric <- plot_data[plot_data$metric == metric, ]
  if (is.null(ylim)) {
    lo <- min(d_metric$mean - d_metric$half, na.rm = TRUE)
    hi <- max(d_metric$mean + d_metric$half, na.rm = TRUE)
    pad <- 0.08 * (hi - lo)
    ylim <- c(max(0, lo - pad), hi + pad)
  }
  if (log_y) {
    ylim <- c(max(1e-3, ylim[1]), ylim[2])
  }
  plot(
    NA, xlim = c(0.85, 4.15), ylim = ylim, log = ifelse(log_y, "y", ""),
    xlab = "Observed CT visits (m)", ylab = ylab,
    xaxt = "n", main = title, cex.main = 0.92
  )
  axis(1, at = 1:4)
  grid(col = "#E6E6E6", lwd = 0.6)
  box(lwd = 0.8)
  for (grp in levels(plot_data$method_group)) {
    d <- d_metric[d_metric$method_group == grp, ]
    d <- d[order(d$m), ]
    lines(d$m, d$mean, col = palette[grp], lwd = 1.35)
    points(d$m, d$mean, col = palette[grp], pch = shape_map[grp], cex = 0.75)
    draw_errorbar(d$m, d$mean - d$half, d$mean + d$half, palette[grp], log_y = log_y)
  }
  if (!is.null(legend_pos)) {
    legend(
      legend_pos, legend = levels(plot_data$method_group),
      col = palette[levels(plot_data$method_group)], pch = shape_map[levels(plot_data$method_group)],
      lty = 1, bty = "n", cex = 0.68, pt.cex = 0.75,
      y.intersp = 0.85, seg.len = 1.2
    )
  }
}

draw_figure <- function() {
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)
  par(
    family = "Times",
    mfrow = c(2, 2),
    mar = c(3.0, 3.35, 1.9, 0.9),
    oma = c(0.2, 0.2, 0.2, 0.2),
    mgp = c(1.8, 0.55, 0),
    tcl = -0.25,
    las = 1,
    cex = 0.75,
    cex.axis = 0.78,
    cex.lab = 0.86
  )

  draw_metric_panel(
    "RMSE", "A. Test RMSE", "RMSE",
    ylim = c(0.40, 1.00), legend_pos = "topright"
  )
  draw_metric_panel(
    "Interval score", "B. Conformal interval score", "Interval score"
  )
  draw_metric_panel(
    "WIS (one level)", "C. Conformal WIS", "WIS"
  )
  draw_metric_panel(
    "Gompertz-style residual", "D. Gompertz-style residual", "Mean absolute residual",
    ylim = c(0.0025, 0.65), log_y = TRUE
  )
}

pdf_files <- c(
  file.path(out_dir, "Figure10.pdf"),
  file.path(out_dir, "fig_experiment4_physics_uq_reliability_cmpb.pdf"),
  file.path(out_dir, "fig_experiment4_physics_uq_reliability.pdf")
)
png_files <- c(
  file.path(out_dir, "Figure10.png"),
  file.path(out_dir, "fig_experiment4_physics_uq_reliability_cmpb.png"),
  file.path(out_dir, "fig_experiment4_physics_uq_reliability.png")
)

for (pdf_file in pdf_files) {
  grDevices::pdf(pdf_file, width = 7.2, height = 4.45, family = "Times", pointsize = 8, useDingbats = FALSE)
  draw_figure()
  dev.off()
}

pdftoppm <- Sys.which("pdftoppm")
if (nzchar(pdftoppm)) {
  for (i in seq_along(pdf_files)) {
    prefix <- sub("\\.png$", "", png_files[i])
    tmp_file <- paste0(prefix, "-1.png")
    system2(pdftoppm, c("-png", "-r", "600", "-singlefile", shQuote(pdf_files[i]), shQuote(prefix)))
    if (file.exists(tmp_file) && tmp_file != png_files[i]) {
      file.rename(tmp_file, png_files[i])
    }
  }
} else {
  warning("pdftoppm not found; PNG previews were not generated. PDF outputs are complete.")
}

cat("Saved Figure 10 outputs to:\n")
cat(paste(c(pdf_files, png_files, source_out), collapse = "\n"))
cat("\n")
