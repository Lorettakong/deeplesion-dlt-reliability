script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", script_args, value = TRUE)
if (length(file_arg)) {
  script_path <- normalizePath(gsub("~\\+~", " ", sub("^--file=", "", file_arg[1])), mustWork = TRUE)
  out_dir <- dirname(script_path)
} else {
  out_dir <- getwd()
}

rmse_path <- file.path(out_dir, "table3a_fixed_lambda_rmse.csv")
resid_path <- file.path(out_dir, "table3b_fixed_lambda_physics_residual.csv")

parse_pm <- function(x) {
  x <- trimws(as.character(x))
  parts <- strsplit(x, "\\s*\\+/-\\s*")
  mean <- as.numeric(vapply(parts, `[`, character(1), 1))
  half <- as.numeric(vapply(parts, function(z) ifelse(length(z) >= 2, z[2], NA_character_), character(1)))
  data.frame(mean = mean, half = half)
}

to_long <- function(path, metric) {
  dat <- read.csv(path, check.names = FALSE)
  lambda_cols <- setdiff(names(dat), "m")
  rows <- lapply(lambda_cols, function(col) {
    parsed <- parse_pm(dat[[col]])
    is_unregularized <- col %in% c("No Regularization", "No Physics")
    lambda_label <- if (is_unregularized) "lambda = 0" else gsub("lambda=", "lambda = ", col)
    lambda_value <- if (is_unregularized) 0 else as.numeric(sub("lambda=", "", col))
    data.frame(
      m = dat$m,
      lambda = lambda_label,
      lambda_value = lambda_value,
      metric = metric,
      mean = parsed$mean,
      half = parsed$half,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

rmse <- to_long(rmse_path, "RMSE")
resid <- to_long(resid_path, "Residual")
plot_data <- merge(
  rmse[, c("m", "lambda", "lambda_value", "mean", "half")],
  resid[, c("m", "lambda", "lambda_value", "mean", "half")],
  by = c("m", "lambda", "lambda_value"),
  suffixes = c("_rmse", "_resid")
)

lambda_levels <- c("lambda = 0", "lambda = 0.1", "lambda = 1", "lambda = 10", "lambda = 100")
palette <- c(
  "lambda = 0" = "#1F77B4",
  "lambda = 0.1" = "#FF7F0E",
  "lambda = 1" = "#2CA02C",
  "lambda = 10" = "#D62728",
  "lambda = 100" = "#9467BD"
)
point_shapes <- c("1" = 16, "2" = 17, "3" = 15, "4" = 18)

save_source <- file.path(out_dir, "figure9_cmpb_source_data.csv")
write.csv(plot_data, save_source, row.names = FALSE)

draw_errorbar <- function(x, y, lo, hi, col, log_y = FALSE, width = 0.045) {
  if (log_y) {
    lo <- pmax(lo, 1e-5)
    hi <- pmax(hi, 1e-5)
  }
  arrows(x, lo, x, hi, angle = 90, code = 3, length = width, col = col, lwd = 0.75)
}

draw_figure <- function() {
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)
  par(
    family = "Times",
    mfrow = c(1, 3),
    mar = c(3.0, 3.15, 1.8, 0.8),
    oma = c(0.2, 0.2, 0.2, 0.2),
    mgp = c(1.8, 0.55, 0),
    tcl = -0.25,
    las = 1,
    cex = 0.75,
    cex.axis = 0.78,
    cex.lab = 0.85
  )

  ## Panel A
  plot(
    NA, xlim = c(0.85, 4.15), ylim = c(0.38, 1.0),
    xlab = "Observed CT visits (m)", ylab = "RMSE",
    xaxt = "n", main = "A. RMSE by regularization weight",
    cex.main = 0.88
  )
  axis(1, at = 1:4)
  grid(col = "#E6E6E6", lwd = 0.6)
  box(lwd = 0.8)
  for (lam in lambda_levels) {
    d <- rmse[rmse$lambda == lam, ]
    ord <- order(d$m)
    d <- d[ord, ]
    lines(d$m, d$mean, col = palette[lam], lwd = 1.4)
    points(d$m, d$mean, col = palette[lam], pch = 16, cex = 0.75)
    draw_errorbar(d$m, d$mean, d$mean - d$half, d$mean + d$half, palette[lam])
  }
  legend(
    "topright", legend = lambda_levels, col = palette[lambda_levels],
    lty = 1, pch = 16, bty = "n", cex = 0.66, pt.cex = 0.65,
    y.intersp = 0.85, seg.len = 1.3
  )

  ## Panel B
  plot(
    NA, xlim = c(0.85, 4.15), ylim = c(1e-3, 0.65), log = "y",
    xlab = "Observed CT visits (m)", ylab = "Mean absolute residual",
    xaxt = "n", main = "B. Gompertz-style residual by regularization weight",
    cex.main = 0.88
  )
  axis(1, at = 1:4)
  grid(col = "#E6E6E6", lwd = 0.6)
  box(lwd = 0.8)
  for (lam in lambda_levels) {
    d <- resid[resid$lambda == lam, ]
    ord <- order(d$m)
    d <- d[ord, ]
    lines(d$m, d$mean, col = palette[lam], lwd = 1.4)
    points(d$m, d$mean, col = palette[lam], pch = 16, cex = 0.75)
    draw_errorbar(d$m, d$mean, d$mean - d$half, d$mean + d$half, palette[lam], log_y = TRUE)
  }

  ## Panel C
  plot(
    NA, xlim = c(0.38, 1.0), ylim = c(1e-3, 0.65), log = "y",
    xlab = "RMSE", ylab = "Mean absolute residual",
    main = "C. Accuracy-residual trade-off", cex.main = 0.88
  )
  grid(col = "#E6E6E6", lwd = 0.6)
  box(lwd = 0.8)
  for (lam in lambda_levels) {
    d <- plot_data[plot_data$lambda == lam, ]
    for (i in seq_len(nrow(d))) {
      points(
        d$mean_rmse[i], d$mean_resid[i],
        pch = point_shapes[as.character(d$m[i])],
        col = palette[lam], bg = palette[lam], cex = 0.8
      )
    }
  }
  legend(
    "topright", legend = lambda_levels, col = palette[lambda_levels],
    pch = 16, bty = "n", cex = 0.62, pt.cex = 0.7, y.intersp = 0.82
  )
  legend(
    "bottomleft", legend = paste0("m = ", 1:4), pch = point_shapes,
    col = "gray30", bty = "n", cex = 0.62, pt.cex = 0.72,
    y.intersp = 0.82
  )
}

pdf_files <- c(
  file.path(out_dir, "Figure9.pdf"),
  file.path(out_dir, "fig_experiment3_physics_tradeoff_cmpb.pdf"),
  file.path(out_dir, "fig_experiment3_physics_tradeoff.pdf")
)
png_files <- c(
  file.path(out_dir, "Figure9.png"),
  file.path(out_dir, "fig_experiment3_physics_tradeoff_cmpb.png"),
  file.path(out_dir, "fig_experiment3_physics_tradeoff.png")
)

for (pdf_file in pdf_files) {
  grDevices::pdf(pdf_file, width = 7.2, height = 2.55, family = "Times", pointsize = 8, useDingbats = FALSE)
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

cat("Saved Figure 9 outputs to:\n")
cat(paste(c(pdf_files, png_files, save_source), collapse = "\n"))
cat("\n")
