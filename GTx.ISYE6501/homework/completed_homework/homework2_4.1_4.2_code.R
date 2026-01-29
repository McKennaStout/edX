rm(list = ls())

i_path <- "C:/Users/mstout/OneDrive - AANP/Documents/Workspace.MAIN/edX/GTx.ISYE6501/data/Homework2_ISYE6501/data 4.2/iris.txt"
i_df <- read.table(i_path, header = TRUE, sep = "", stringsAsFactors = FALSE)

dim(i_df)
head(i_df, 16)
names(i_df)
i_y_true <- i_df$Species
i_x_all  <- i_df[, c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width")]
i_x_all_scaled <- scale(i_x_all)
i_majority_map_predict <- function(i_cluster_id, i_y) {
  i_tab <- table(i_cluster_id, i_y)
  i_winners <- apply(i_tab, 1, function(row) names(row)[which.max(row)])
  i_y_pred <- i_winners[as.character(i_cluster_id)]
  return(i_y_pred)
}
i_score_kmeans <- function(i_x_scaled, i_y, i_k, nstart = 50, seed = 16) {
  set.seed(seed)
  i_fit <- kmeans(i_x_scaled, centers = i_k, nstart = nstart)
  i_y_pred <- i_majority_map_predict(i_fit$cluster, i_y)
  i_acc <- mean(i_y_pred == i_y)
  list(acc = i_acc, fit = i_fit)
}
i_predictor_names <- colnames(i_x_all_scaled)
i_subsets <- unlist(
  lapply(1:length(i_predictor_names),
         function(m) combn(i_predictor_names, m, simplify = FALSE)),
  recursive = FALSE
)
i_k_grid <- 2:6
i_results <- data.frame(
  predictors = character(),
  k = integer(),
  accuracy = numeric(),
  tot_withinss = numeric(),
  stringsAsFactors = FALSE
)
i_best <- list(acc = -Inf, predictors = NULL, k = NA, fit = NULL)
for (i_vars in i_subsets) {
  i_x_sub <- i_x_all_scaled[, i_vars, drop = FALSE]

  for (i_k in i_k_grid) {
    i_out <- i_score_kmeans(i_x_sub, i_y_true, i_k = i_k, nstart = 50)

    i_results <- rbind(
      i_results,
      data.frame(
        predictors = paste(i_vars, collapse = ", "),
        k = i_k,
        accuracy = i_out$acc,
        tot_withinss = i_out$fit$tot.withinss,
        stringsAsFactors = FALSE
      )
    )

    if (i_out$acc > i_best$acc) {
      i_best$acc <- i_out$acc
      i_best$predictors <- i_vars
      i_best$k <- i_k
      i_best$fit <- i_out$fit
    }
  }
}
i_results <- i_results[order(-i_results$accuracy, i_results$tot_withinss), ]
print(head(i_results, 10))

cat("\nBEST SETUP\n")
cat("Predictors:", paste(i_best$predictors, collapse = ", "), "\n")
cat("k:", i_best$k, "\n")
cat("Accuracy:", round(i_best$acc, 4), "\n\n")
i_x_best <- i_x_all_scaled[, i_best$predictors, drop = FALSE]
i_y_best_pred <- i_majority_map_predict(i_best$fit$cluster, i_y_true)

print(table(Predicted = i_y_best_pred, Actual = i_y_true))
if (ncol(i_x_best) == 1) {

  opar <- par(no.readonly = TRUE)
  on.exit(par(opar), add = TRUE)

  layout(matrix(c(1, 2), nrow = 2), heights = c(4, 1))

  par(mar = c(4, 4, 3, 1))
  plot(
    seq_along(i_x_best[, 1]),
    i_x_best[, 1],
    col = i_best$fit$cluster,
    pch = 19,
    xlab = "Observation index",
    ylab = i_best$predictors[1],
    main = paste("kmeans clusters (k =", i_best$k, ")")
  )

  i_mis <- which(i_y_best_pred != i_y_true)
  points(i_mis, i_x_best[i_mis, 1], pch = 1, cex = 2, lwd = 2)

  i_cluster_species <- tapply(i_y_best_pred, i_best$fit$cluster, function(x) x[1])

  par(mar = c(0, 0, 0, 0))
  plot.new()
  legend(
    "center",
    legend = i_cluster_species,
    col = as.numeric(names(i_cluster_species)),
    pch = 19,
    title = "Species Names",
    horiz = TRUE,
    bty = "n",
    cex = 0.9
  )

  layout(1)
}


opar <- par(no.readonly = TRUE)
on.exit(par(opar), add = TRUE)

i_pw_species_mean <- ave(i_df$Petal.Width, i_df$Species, FUN = mean)
i_gene <- i_df$Petal.Width > i_pw_species_mean

i_species_gene <- ifelse(
  i_gene,
  paste(i_df$Species, "w/ gene"),
  as.character(i_df$Species)
)

i_species_gene <- factor(
  i_species_gene,
  levels = c("setosa", "setosa w/ gene",
             "versicolor", "versicolor w/ gene",
             "virginica", "virginica w/ gene")
)

set.seed(16)
i_pw_scaled <- as.numeric(scale(i_df$Petal.Width))
i_k6_fit <- kmeans(i_pw_scaled, centers = 6, nstart = 50)

i_cluster_letters <- LETTERS[i_k6_fit$cluster]

i_centers <- sort(as.numeric(i_k6_fit$centers))
i_bounds_scaled <- (i_centers[-1] + i_centers[-length(i_centers)]) / 2
i_pw_center <- attr(scale(i_df$Petal.Width), "scaled:center")
i_pw_scale  <- attr(scale(i_df$Petal.Width), "scaled:scale")
i_bounds_pw <- i_bounds_scaled * i_pw_scale + i_pw_center

layout(matrix(c(1, 2), nrow = 2), heights = c(4, 1))

par(mar = c(4, 4, 3, 1))
plot(
  seq_along(i_df$Petal.Width),
  i_df$Petal.Width,
  type = "n",
  xlab = "Observation index",
  ylab = "Petal.Width",
  main = "kmeans k=6 on Petal.Width"
)

abline(h = i_bounds_pw, lty = 2)

text(
  seq_along(i_df$Petal.Width),
  i_df$Petal.Width,
  labels = i_cluster_letters,
  col = as.integer(i_species_gene),
  cex = 0.9
)

i_mis <- which(i_y_best_pred != i_y_true)
points(i_mis, i_df$Petal.Width[i_mis], pch = 1, cex = 2, lwd = 2)

par(mar = c(0, 0, 0, 0))
plot.new()

legend(
  "top",
  legend = levels(i_species_gene),
  col = seq_along(levels(i_species_gene)),
  pch = 19,
  title = "Species / gene",
  horiz = TRUE,
  bty = "n",
  cex = 0.85
)

legend(
  "bottom",
  legend = paste("Cluster", LETTERS[1:6]),
  title = "kmeans (k=6)",
  horiz = TRUE,
  bty = "n",
  cex = 0.85
)

layout(1)
