rm(list=ls()) # deletes everything from memory

library(kknn)
library(caret)
library(kernlab)   # SVM (ksvm)

set.seed(16) # do not change

df <- read.delim(
  "C:/Users/mstout/OneDrive - AANP/Documents/Workspace.MAIN/edX/GTx.ISYE6501/data/Homework2_ISYE6501/data 3.1/credit_card_data-headers.txt",
  header = TRUE
)

print(dim(df)) # confirm data loaded

# ------------------------------------------------------------
# Basic prep
# ------------------------------------------------------------

# Response as factor (classification)
df$R1 <- factor(df$R1, levels = c(0,1))

# Binary predictors as factors (helps formula handling; SVM will one-hot encode later)
bin_cols <- c("A1","A9","A10","A11","A12")
bin_cols <- intersect(bin_cols, names(df))
for (cc in bin_cols) df[[cc]] <- factor(df[[cc]])

# Predictor column names
x_cols <- setdiff(names(df), "R1")

# KNN formula: R1 ~ all other columns
form <- as.formula(
  paste("R1 ~", paste(x_cols, collapse = " + "))
)

# ------------------------------------------------------------
# (1) KNN with 10-fold Cross-Validation over k  [REQUIRED]
#   - Global scaling is OK for CV demonstration
# ------------------------------------------------------------

# Copy df for KNN-CV scaling (so we can keep an unscaled version for later splits)
df_knn_cv <- df

num_cols <- names(df_knn_cv)[sapply(df_knn_cv, is.numeric)]
num_cols <- setdiff(num_cols, "R1")
df_knn_cv[num_cols] <- scale(df_knn_cv[num_cols])

K <- 10
fold_id <- sample(rep(1:K, length.out = nrow(df_knn_cv)))

k_grid <- 1:50

cv_acc <- sapply(k_grid, function(k){
  cat("CV (KNN) k =", k, "\n")

  fold_acc <- sapply(1:K, function(f){
    tr <- df_knn_cv[fold_id != f, , drop = FALSE]
    te <- df_knn_cv[fold_id == f, , drop = FALSE]

    fit  <- kknn(form, train = tr, test = te,
                 k = k, distance = 2,
                 kernel = "optimal", scale = FALSE)
    pred <- fitted(fit)
    mean(pred == te$R1)
  })

  mean(fold_acc)
})

best_k_cv <- k_grid[which.max(cv_acc)]
print(best_k_cv)
print(max(cv_acc))

# In-sample (reporting only; not out-of-sample)
fit_cv_final <- kknn(form, train = df_knn_cv, test = df_knn_cv,
                     k = best_k_cv, distance = 2,
                     kernel = "optimal", scale = FALSE)
cv_in_sample_acc <- mean(fitted(fit_cv_final) == df_knn_cv$R1)
print(cv_in_sample_acc)

# ------------------------------------------------------------
# (2) Train / Validation / Test split (KNN)       [REQUIRED]
#   - Proper scaling: fit scaler on TRAIN only
# ------------------------------------------------------------

set.seed(42)

idx_train <- createDataPartition(df$R1, p = 0.60, list = FALSE)
train_df <- df[idx_train, , drop = FALSE]
temp_df  <- df[-idx_train, , drop = FALSE]

idx_valid <- createDataPartition(temp_df$R1, p = 0.50, list = FALSE)
valid_df <- temp_df[idx_valid, , drop = FALSE]
test_df  <- temp_df[-idx_valid, , drop = FALSE]

# Fit scaling on TRAIN numeric columns only (prevents leakage)
num_cols2 <- names(train_df)[sapply(train_df, is.numeric)]
num_cols2 <- setdiff(num_cols2, "R1")

sc <- preProcess(train_df[, num_cols2, drop = FALSE], method = c("center","scale"))

train_knn <- train_df
valid_knn <- valid_df
test_knn  <- test_df

train_knn[, num_cols2] <- predict(sc, train_knn[, num_cols2, drop = FALSE])
valid_knn[, num_cols2] <- predict(sc, valid_knn[, num_cols2, drop = FALSE])
test_knn[,  num_cols2] <- predict(sc, test_knn[,  num_cols2, drop = FALSE])

k_grid2 <- 1:50

val_acc <- sapply(k_grid2, function(k){
  cat("Validation (KNN) k =", k, "\n")
  fit  <- kknn(form, train = train_knn, test = valid_knn,
               k = k, distance = 2,
               kernel = "optimal", scale = FALSE)
  pred <- fitted(fit)
  mean(pred == valid_knn$R1)
})

best_k_val <- k_grid2[which.max(val_acc)]
print(best_k_val)
print(max(val_acc))

trainval_knn <- rbind(train_knn, valid_knn)

fit_test <- kknn(form, train = trainval_knn, test = test_knn,
                 k = best_k_val, distance = 2,
                 kernel = "optimal", scale = FALSE)

pred_test <- fitted(fit_test)

acc_test <- mean(pred_test == test_knn$R1)
print(acc_test)
print(confusionMatrix(pred_test, test_knn$R1))

# ------------------------------------------------------------
# (3) SVM with 10-fold Cross-Validation (linear + RBF) [OPTIONAL]
#   - Use model.matrix() so X is numeric (one-hot encode factors)
#   - Use scaled=TRUE inside ksvm per course guidance
# ------------------------------------------------------------

# Build numeric design matrix for FULL df (one-hot encoding)
X_all <- model.matrix(~ . - 1, data = df[, x_cols, drop = FALSE])
y_all <- df$R1

C_grid <- c(0.001, 0.01, 0.1, 1, 10, 100, 1000)

# Linear SVM CV
svm_cv_acc <- sapply(C_grid, function(Cval){
  cat("CV (SVM linear) C =", Cval, "\n")

  fold_acc <- sapply(1:K, function(f){
    tr_idx <- which(fold_id != f)
    te_idx <- which(fold_id == f)

    X_tr <- X_all[tr_idx, , drop = FALSE]
    y_tr <- y_all[tr_idx]

    X_te <- X_all[te_idx, , drop = FALSE]
    y_te <- y_all[te_idx]

    mdl <- ksvm(
      x = X_tr,
      y = y_tr,
      type = "C-svc",
      kernel = "vanilladot",
      C = Cval,
      scaled = TRUE
    )

    pred <- predict(mdl, X_te)
    mean(pred == y_te)
  })

  mean(fold_acc)
})

best_C_cv <- C_grid[which.max(svm_cv_acc)]
print(best_C_cv)
print(max(svm_cv_acc))

# RBF SVM CV (nonlinear)
sigma_est <- as.numeric(sigest(X_all)[2])

svm_rbf_cv_acc <- sapply(C_grid, function(Cval){
  cat("CV (SVM RBF) C =", Cval, "sigma =", sigma_est, "\n")

  fold_acc <- sapply(1:K, function(f){
    tr_idx <- which(fold_id != f)
    te_idx <- which(fold_id == f)

    X_tr <- X_all[tr_idx, , drop = FALSE]
    y_tr <- y_all[tr_idx]

    X_te <- X_all[te_idx, , drop = FALSE]
    y_te <- y_all[te_idx]

    mdl <- ksvm(
      x = X_tr,
      y = y_tr,
      type = "C-svc",
      kernel = rbfdot(sigma = sigma_est),
      C = Cval,
      scaled = TRUE
    )

    pred <- predict(mdl, X_te)
    mean(pred == y_te)
  })

  mean(fold_acc)
})

best_C_rbf_cv <- C_grid[which.max(svm_rbf_cv_acc)]
print(best_C_rbf_cv)
print(max(svm_rbf_cv_acc))

# ------------------------------------------------------------
# (4) SVM with Train / Validation / Test split (linear + RBF) [OPTIONAL]
#   - Use the SAME split already created above
#   - Tune C on validation; evaluate once on test
# ------------------------------------------------------------

# Build numeric matrices for split data (one-hot encoding)
X_train <- model.matrix(~ . - 1, data = train_df[, x_cols, drop = FALSE])
y_train <- train_df$R1

X_valid <- model.matrix(~ . - 1, data = valid_df[, x_cols, drop = FALSE])
y_valid <- valid_df$R1

X_testm <- model.matrix(~ . - 1, data = test_df[, x_cols, drop = FALSE])
y_test  <- test_df$R1

# Linear SVM validation tuning
svm_val_acc <- sapply(C_grid, function(Cval){
  cat("Validation (SVM linear) C =", Cval, "\n")

  mdl <- ksvm(
    x = X_train,
    y = y_train,
    type = "C-svc",
    kernel = "vanilladot",
    C = Cval,
    scaled = TRUE
  )

  pred <- predict(mdl, X_valid)
  mean(pred == y_valid)
})

best_C_val <- C_grid[which.max(svm_val_acc)]
print(best_C_val)
print(max(svm_val_acc))

# Refit linear SVM on Train+Valid, test once
X_trainval <- rbind(X_train, X_valid)
y_trainval <- c(y_train, y_valid)

svm_trainval <- ksvm(
  x = X_trainval,
  y = y_trainval,
  type = "C-svc",
  kernel = "vanilladot",
  C = best_C_val,
  scaled = TRUE
)

svm_pred_test <- predict(svm_trainval, X_testm)
svm_acc_test <- mean(svm_pred_test == y_test)
print(svm_acc_test)
print(confusionMatrix(svm_pred_test, y_test))

# RBF SVM validation tuning
svm_rbf_val_acc <- sapply(C_grid, function(Cval){
  cat("Validation (SVM RBF) C =", Cval, "sigma =", sigma_est, "\n")

  mdl <- ksvm(
    x = X_train,
    y = y_train,
    type = "C-svc",
    kernel = rbfdot(sigma = sigma_est),
    C = Cval,
    scaled = TRUE
  )

  pred <- predict(mdl, X_valid)
  mean(pred == y_valid)
})

best_C_rbf_val <- C_grid[which.max(svm_rbf_val_acc)]
print(best_C_rbf_val)
print(max(svm_rbf_val_acc))

# Refit RBF SVM on Train+Valid, test once
svm_rbf_trainval <- ksvm(
  x = X_trainval,
  y = y_trainval,
  type = "C-svc",
  kernel = rbfdot(sigma = sigma_est),
  C = best_C_rbf_val,
  scaled = TRUE
)

svm_rbf_pred_test <- predict(svm_rbf_trainval, X_testm)
svm_rbf_acc_test <- mean(svm_rbf_pred_test == y_test)
print(svm_rbf_acc_test)
print(confusionMatrix(svm_rbf_pred_test, y_test))
