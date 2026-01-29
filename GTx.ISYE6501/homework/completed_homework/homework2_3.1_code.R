 rm(list=ls())  # Uncomment only if starting fresh

library(kknn)
library(caret)
library(kernlab)

set.seed(16)  # Keep fixed for reproducibility, I chose 16 because it's my favorite number
df <- read.delim(
  "C:/Users/mstout/OneDrive - AANP/Documents/Workspace.MAIN/edX/GTx.ISYE6501/data/Homework2_ISYE6501/data 3.1/credit_card_data-headers.txt",
  header = TRUE
)

dim(df) # forcing it to show me the dimensions of the table
head(df, 16) # forcing it to show me the first 16 rows to ensure data quality

df$R1 <- factor(df$R1, levels = c(0, 1))
table(df$R1)
bin_cols <- c("A1","A9","A10","A11","A12")
bin_cols <- intersect(bin_cols, names(df))

for (cc in bin_cols) {
  df[[cc]] <- factor(df[[cc]])
}

str(df)
num_cols <- names(df)[sapply(df, is.numeric)]
num_cols <- setdiff(num_cols, "R1")

df[num_cols] <- scale(df[num_cols])
form <- as.formula(
  paste("R1 ~", paste(setdiff(names(df), "R1"), collapse = " + "))
)

K <- 10
fold_id <- sample(rep(1:K, length.out = nrow(df)))
k_grid <- 1:21

cv_acc <- sapply(k_grid, function(k) {

  fold_acc <- sapply(1:K, function(f) {
    tr <- df[fold_id != f, , drop = FALSE]
    te <- df[fold_id == f, , drop = FALSE]

    fit <- kknn(
      form,
      train = tr, # training
      test  = te, # testing
      k = k,
      distance = 2,
      scale = FALSE
    )

    pred <- fitted(fit)
    mean(pred == te$R1)
  })

  mean(fold_acc)
})

best_k_cv <- k_grid[which.max(cv_acc)]
best_k_cv
max(cv_acc)


set.seed(16) # Keep fixed for reproducibility, I chose 16 because it's my favorite number

idx_train <- createDataPartition(df$R1, p = 0.60, list = FALSE)
train_df <- df[idx_train, , drop = FALSE]
temp_df  <- df[-idx_train, , drop = FALSE]

idx_valid <- createDataPartition(temp_df$R1, p = 0.50, list = FALSE)
valid_df <- temp_df[idx_valid, , drop = FALSE]
test_df  <- temp_df[-idx_valid, , drop = FALSE]

dim(train_df); dim(valid_df); dim(test_df)
num_cols2 <- names(train_df)[sapply(train_df, is.numeric)]
num_cols2 <- setdiff(num_cols2, "R1")

sc <- preProcess(train_df[, num_cols2, drop = FALSE],
                 method = c("center", "scale"))

train_df[, num_cols2] <- predict(sc, train_df[, num_cols2])
valid_df[, num_cols2] <- predict(sc, valid_df[, num_cols2])
test_df[,  num_cols2] <- predict(sc, test_df[,  num_cols2])
k_grid2 <- 1:21

val_acc <- sapply(k_grid2, function(k) {

  fit <- kknn(
    form,
    train = train_df,
    test  = valid_df,
    k = k,
    distance = 2,
    scale = FALSE
  )

  pred <- fitted(fit)
  mean(pred == valid_df$R1)
})

best_k_val <- k_grid2[which.max(val_acc)]
best_k_val
max(val_acc)
trainval_df <- rbind(train_df, valid_df)

fit_test <- kknn(
  form,
  train = trainval_df,
  test  = test_df,
  k = best_k_val, # <-- this is 20
  distance = 2,
  scale = FALSE
)

pred_test <- fitted(fit_test)
cm <- caret::confusionMatrix(pred_test, test_df$R1) # unless you want to see a bunch of data you don't really need...
cm$overall["Accuracy"] # only pull...
cm$table # what you need!
fp_test <- cm$table["1", "0"]
n_test  <- sum(cm$table)
n_total <- nrow(df)

fp_test / n_test * n_total

X <- model.matrix(R1 ~ . - 1, data = df)
y <- df$R1

dim(X)
C_grid <- c(0.001, 0.01, 0.1, 1, 10, 100, 1000)

svm_cv_acc <- sapply(C_grid, function(Cval) {

  fold_acc <- sapply(1:K, function(f) {
    tr_idx <- which(fold_id != f)
    te_idx <- which(fold_id == f)

    Xtr <- X[tr_idx, , drop = FALSE]
    ytr <- y[tr_idx]
    Xte <- X[te_idx, , drop = FALSE]
    yte <- y[te_idx]

    nzv <- apply(Xtr, 2, var) > 0
    Xtr <- Xtr[, nzv, drop = FALSE]
    Xte <- Xte[, nzv, drop = FALSE]

    pp <- suppressWarnings(preProcess(Xtr, method = c("center","scale")))
    Xtr_s <- predict(pp, Xtr)
    Xte_s <- predict(pp, Xte)

    model <- suppressWarnings(
      suppressMessages(
        ksvm(
          x = Xtr_s,
          y = ytr,
          type = "C-svc",
          kernel = "vanilladot",
          C = Cval,
          scaled = FALSE
        )
      )
    )

    pred <- predict(model, Xte_s)
    mean(pred == yte)
  })

  mean(fold_acc)
})

best_C_linear <- C_grid[which.max(svm_cv_acc)]
best_acc_linear <- max(svm_cv_acc)
best_C_linear
best_acc_linear
