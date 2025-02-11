# ===============================
# Ładowanie bibliotek
# ===============================
library(readxl)
library(ggplot2)
library(rpart)
library(caret)
library(nnet)
library(gridExtra)
library(reshape2)

# ============================================================
# Problem Regresja
# ============================================================

source("funkcje.R")

data <- read_excel("Concrete_Data.xls")
colnames(data) <- c("Cement", "Blast_Furnace_Slag", "Fly_Ash", "Water", 
                    "Superplasticizer", "Coarse_Aggregate", "Fine_Aggregate", 
                    "Age", "Concrete_Compressive_Strength")
print(colnames(data))

target <- data$Concrete_Compressive_Strength
features <- data[, -ncol(data)]
features_scaled <- as.data.frame(scale(features))
data_reg <- cbind(features_scaled, Concrete_Compressive_Strength = target)

set.seed(123)

# Definicja zakresu wartości dla hiperparametru k
k_values <- seq(3, 15, by = 2)

results_knn_custom <- data.frame(k = numeric(), rmse = numeric(), r2 = numeric())

# Iteracja po różnych wartościach k
for (k_val in k_values) {
  hyperparams_knn_custom <- list(k = k_val, threshold = 0.5)
  
  # Przeprowadzenie 5-krotnej CV
  cv_results <- custom_cv(
    data = data_reg,
    k_folds = 5,
    train_func = train_knn,
    predict_func = predict_knn,
    performance_funcs = list(rmse = rmse, r2 = r2_score),
    hyperparams = hyperparams_knn_custom,
    target_col = "Concrete_Compressive_Strength",
    pass_to_predict = TRUE
  )
  
  # Zapisujemy średnie metryki dla części testowej CV
  results_knn_custom <- rbind(results_knn_custom, 
                              data.frame(k = k_val,
                                         rmse = cv_results$test$rmse$mean,
                                         r2 = cv_results$test$r2$mean))
}

print(results_knn_custom)

# Wybór najlepszego modelu na podstawie minimalnego RMSE
best_knn_model <- results_knn_custom[which.min(results_knn_custom$rmse), ]
print(best_knn_model)

cat("Najlepsza konfiguracja modelu k-NN (regresja) na podstawie RMSE:\n")
cat("Liczba najbliższych sąsiadów (k):", best_knn_model$k, "\n")
cat("RMSE:", best_knn_model$rmse, "\n")
cat("R^2:", best_knn_model$r2, "\n")

# Wizualizacja zależności RMSE od liczby sąsiadów k
ggplot(results_knn_custom, aes(x = k, y = rmse)) +
  geom_line() +
  geom_point() +
  ggtitle("Własny model k-NN: RMSE w zależności od k") +
  xlab("Liczba najbliższych sąsiadów (k)") +
  ylab("RMSE")

# Wizualizacja zależności R^2 od liczby sąsiadów k
ggplot(results_knn_custom, aes(x = k, y = r2)) +
  geom_line(color = "blue") +
  geom_point(color = "blue") +
  ggtitle("Własny model k-NN: R^2 w zależności od k") +
  xlab("Liczba najbliższych sąsiadów (k)") +
  ylab(expression(R^2))


# Definicja zakresu wartości dla hiperparametru max_depth
depth_values <- c(3, 5, 7, 9, 11)

results_tree_custom <- data.frame(max_depth = numeric(), rmse = numeric(), r2 = numeric())

# Iteracja po różnych wartościach max_depth
for (depth_val in depth_values) {
  hyperparams_tree_custom <- list(max_depth = depth_val)
  
  # Przeprowadzenie 5-krotnej CV
  cv_results_tree <- custom_cv(
    data = data_reg,
    k_folds = 5,
    train_func = train_tree,
    predict_func = predict_tree,
    performance_funcs = list(rmse = rmse, r2 = r2_score),
    hyperparams = hyperparams_tree_custom,
    target_col = "Concrete_Compressive_Strength",
    pass_to_predict = FALSE
  )
  
  # Zapisujemy średnie metryki dla części testowej CV
  results_tree_custom <- rbind(results_tree_custom,
                               data.frame(max_depth = depth_val,
                                          rmse = cv_results_tree$test$rmse$mean,
                                          r2 = cv_results_tree$test$r2$mean))
}

print(results_tree_custom)

# Wybór najlepszego modelu na podstawie minimalnego RMSE
best_tree_model <- results_tree_custom[which.min(results_tree_custom$rmse), ]
print(best_tree_model)

cat("Najlepsza konfiguracja drzewa decyzyjnego (regresja):\n")
cat("Maksymalna głębokość:", best_tree_model$max_depth, "\n")
cat("RMSE:", best_tree_model$rmse, "\n")
cat("R^2:", best_tree_model$r2, "\n")

# Wizualizacja wyników – zależność RMSE od max_depth
ggplot(results_tree_custom, aes(x = max_depth, y = rmse)) +
  geom_line() +
  geom_point() +
  ggtitle("Własny model drzewa decyzyjnego: RMSE w zależności od max_depth") +
  xlab("Maksymalna głębokość drzewa (max_depth)") +
  ylab("RMSE")

# Wizualizacja wyników – zależność R^2 od max_depth
ggplot(results_tree_custom, aes(x = max_depth, y = r2)) +
  geom_line(color = "blue") +
  geom_point(color = "blue") +
  ggtitle("Własny model drzewa decyzyjnego: R^2 w zależności od max_depth") +
  xlab("Maksymalna głębokość drzewa (max_depth)") +
  ylab(expression(R^2))


# Definicja siatki hiperparametrów dla sieci neuronowej
nn_grid <- expand.grid(hidden_neurons = c(10, 30, 50, 70, 100),
                       learning_rate = c(0.0001, 0.001, 0.01))

results_nn_custom <- data.frame(hidden_neurons = numeric(),
                                learning_rate = numeric(),
                                rmse = numeric(),
                                r2 = numeric())

# Iteracja po każdej kombinacji hiperparametrów
for (i in 1:nrow(nn_grid)) {
  h_neurons <- nn_grid$hidden_neurons[i]
  lr <- nn_grid$learning_rate[i]
  
  hyperparams_nn_custom <- list(hidden_neurons = h_neurons,
                                epochs = 2000,
                                learning_rate = lr)
  
  # Przeprowadzenie 5-krotnej CV
  cv_results_nn <- custom_cv(
    data = data_reg,
    k_folds = 5,
    train_func = train_nn,
    predict_func = predict_nn,
    performance_funcs = list(rmse = rmse, r2 = r2_score),
    hyperparams = hyperparams_nn_custom,
    target_col = "Concrete_Compressive_Strength",
    pass_to_predict = FALSE
  )
  
  # Zapisujemy średnie metryki dla części testowej CV
  results_nn_custom <- rbind(results_nn_custom,
                             data.frame(hidden_neurons = h_neurons,
                                        learning_rate = lr,
                                        rmse = cv_results_nn$test$rmse$mean,
                                        r2 = cv_results_nn$test$r2$mean))
}

print(results_nn_custom)

# Wybór najlepszego modelu na podstawie minimalnego RMSE
best_nn_model <- results_nn_custom[which.min(results_nn_custom$rmse), ]
print(best_nn_model)

cat("Najlepsza konfiguracja sieci neuronowej (regresja):\n")
cat("Liczba neuronów:", best_nn_model$hidden_neurons, "\n")
cat("Learning rate:", best_nn_model$learning_rate, "\n")
cat("RMSE:", best_nn_model$rmse, "\n")
cat("R^2:", best_nn_model$r2, "\n")

# Wizualizacja wyników – RMSE w zależności od liczby neuronów i współczynnika uczenia
ggplot(results_nn_custom, aes(x = factor(hidden_neurons), y = rmse, fill = factor(learning_rate))) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Własny model sieci neuronowej: RMSE") +
  xlab("Liczba neuronów (hidden_neurons)") +
  ylab("RMSE") +
  labs(fill = "Learning Rate")

# Wizualizacja wyników – R^2 w zależności od liczby neuronów i współczynnika uczenia
ggplot(results_nn_custom, aes(x = factor(hidden_neurons), y = r2, fill = factor(learning_rate))) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Własny model sieci neuronowej: R^2") +
  xlab("Liczba neuronów (hidden_neurons)") +
  ylab(expression(R^2)) +
  labs(fill = "Learning Rate")


hyperparams_knn <- list(k = best_knn_model$k)
hyperparams_tree <- list(max_depth = best_tree_model$max_depth)
hyperparams_nn   <- list(hidden_neurons = best_nn_model$hidden_neurons, epochs = 2000, learning_rate = best_nn_model$learning_rate)

metrics_knn <- custom_cv(
  data = data_reg,
  k_folds = 5,
  train_func = train_knn,
  predict_func = predict_knn,
  performance_funcs = list(rmse = rmse, r2 = r2_score),
  hyperparams = hyperparams_knn,
  target_col = "Concrete_Compressive_Strength",
  pass_to_predict = TRUE
)

metrics_tree <- custom_cv(
  data = data_reg,
  k_folds = 5,
  train_func = train_tree,
  predict_func = predict_tree,
  performance_funcs = list(rmse = rmse, r2 = r2_score),
  hyperparams = hyperparams_tree,
  target_col = "Concrete_Compressive_Strength",
  pass_to_predict = FALSE
)

metrics_nn <- custom_cv(
  data = data_reg,
  k_folds = 5,
  train_func = train_nn,
  predict_func = predict_nn,
  performance_funcs = list(rmse = rmse, r2 = r2_score),
  hyperparams = hyperparams_nn,
  target_col = "Concrete_Compressive_Strength",
  pass_to_predict = FALSE
)

cat("Wyniki CV dla własnych modeli:\n")
cat("----------------------------------------------------\n")

# Wyniki dla modelu k-NN
cat("Model k-NN (Regresja):\n")
cat("Trening:\n")
cat("  RMSE:", metrics_knn$train$rmse$mean, "(sd:", metrics_knn$train$rmse$sd, ")\n")
cat("  R^2:  ", metrics_knn$train$r2$mean, "(sd:", metrics_knn$train$r2$sd, ")\n\n")
cat("Walidacja (Test):\n")
cat("  RMSE:", metrics_knn$test$rmse$mean, "(sd:", metrics_knn$test$rmse$sd, ")\n")
cat("  R^2:  ", metrics_knn$test$r2$mean, "(sd:", metrics_knn$test$r2$sd, ")\n\n")
cat("Różnica (Test - Trening):\n")
cat("  RMSE diff:", metrics_knn$diff$rmse$mean_diff, "\n")
cat("  R^2 diff:  ", metrics_knn$diff$r2$mean_diff, "\n")
cat("----------------------------------------------------\n\n")


# Wyniki dla modelu drzewa decyzyjnego
cat("Model drzewa decyzyjnego (Regresja):\n")
cat("Trening:\n")
cat("  RMSE:", metrics_tree$train$rmse$mean, "(sd:", metrics_tree$train$rmse$sd, ")\n")
cat("  R^2:  ", metrics_tree$train$r2$mean, "(sd:", metrics_tree$train$r2$sd, ")\n\n")
cat("Walidacja (Test):\n")
cat("  RMSE:", metrics_tree$test$rmse$mean, "(sd:", metrics_tree$test$rmse$sd, ")\n")
cat("  R^2:  ", metrics_tree$test$r2$mean, "(sd:", metrics_tree$test$r2$sd, ")\n\n")
cat("Różnica (Test - Trening):\n")
cat("  RMSE diff:", metrics_tree$diff$rmse$mean_diff, "\n")
cat("  R^2 diff:  ", metrics_tree$diff$r2$mean_diff, "\n")
cat("----------------------------------------------------\n\n")


# Wyniki dla modelu sieci neuronowej
cat("Model sieci neuronowej (Regresja):\n")
cat("Trening:\n")
cat("  RMSE:", metrics_nn$train$rmse$mean, "(sd:", metrics_nn$train$rmse$sd, ")\n")
cat("  R^2:  ", metrics_nn$train$r2$mean, "(sd:", metrics_nn$train$r2$sd, ")\n\n")
cat("Walidacja (Test):\n")
cat("  RMSE:", metrics_nn$test$rmse$mean, "(sd:", metrics_nn$test$rmse$sd, ")\n")
cat("  R^2:  ", metrics_nn$test$r2$mean, "(sd:", metrics_nn$test$r2$sd, ")\n\n")
cat("Różnica (Test - Trening):\n")
cat("  RMSE diff:", metrics_nn$diff$rmse$mean_diff, "\n")
cat("  R^2 diff:  ", metrics_nn$diff$r2$mean_diff, "\n")
cat("----------------------------------------------------\n")


# Porównanie z modelami z pakietów
ctrl <- trainControl(method = "cv", number = 5)

# Model KNN przy użyciu caret
knn_model_caret <- train(Concrete_Compressive_Strength ~ ., 
                         data = data_reg, 
                         method = "knn", 
                         tuneGrid = data.frame(k = hyperparams_knn$k),
                         trControl = ctrl)
caret_knn_rmse <- min(knn_model_caret$results$RMSE)
caret_knn_r2   <- min(knn_model_caret$results$Rsquared)

# Model drzewa decyzyjnego przy użyciu rpart
tree_model_caret <- train(Concrete_Compressive_Strength ~ ., 
                          data = data_reg, 
                          method = "rpart",
                          tuneGrid = data.frame(cp = 0),
                          trControl = ctrl,
                          control = rpart.control(maxdepth = hyperparams_tree$max_depth))
caret_tree_rmse <- min(tree_model_caret$results$RMSE)
caret_tree_r2   <- max(tree_model_caret$results$Rsquared)

# Model sieci neuronowej przy użyciu nnet
nn_model_caret <- train(Concrete_Compressive_Strength ~ ., 
                        data = data_reg, 
                        method = "nnet", 
                        tuneGrid = data.frame(size = hyperparams_nn$hidden_neurons, decay = hyperparams_nn$learning_rate),
                        trControl = ctrl,
                        maxit = hyperparams_nn$epochs,
                        linout = TRUE,
                        trace = FALSE,
                        MaxNWts = 2000)
caret_nn_rmse <- min(nn_model_caret$results$RMSE)
caret_nn_r2   <- max(nn_model_caret$results$Rsquared)

cat("\nWyniki modeli pakietowych:\n")
cat("Caret KNN RMSE:", caret_knn_rmse, "R^2:", caret_knn_r2, "\n")
cat("Caret Drzewo RMSE:", caret_tree_rmse, "R^2:", caret_tree_r2, "\n")
cat("Caret Sieć NN RMSE:", caret_nn_rmse, "R^2:", caret_nn_r2, "\n")

results_comparison_cv <- data.frame(
  Model = rep(c("KNN", "Drzewo Decyzyjne", "Sieć Neuronowa"), 2),
  Podejście = rep(c("Własne", "Pakietowe"), each = 3),
  RMSE = c(metrics_knn$test$rmse$mean, metrics_tree$test$rmse$mean, metrics_nn$test$rmse$mean, 
           caret_knn_rmse, caret_tree_rmse, caret_nn_rmse),
  R2   = c(metrics_knn$test$r2$mean, metrics_tree$test$r2$mean, metrics_nn$test$r2$mean,
           caret_knn_r2, caret_tree_r2, caret_nn_r2)
)

print(results_comparison_cv)

# Wykres dla RMSE
ggplot(results_comparison_cv, aes(x = Model, y = RMSE, fill = Podejście)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Porównanie RMSE: Własne vs. Pakietowe") +
  theme_minimal()

# Wykres dla R^2
ggplot(results_comparison_cv, aes(x = Model, y = R2, fill = Podejście)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Porównanie R^2: Własne vs. Pakietowe") +
  theme_minimal()

# ============================================================
# Problem Klasyfikacji binarnej
# ============================================================

source("funkcje.R")

data <- read.csv("bank.csv", sep = ";")

data$y <- ifelse(data$y == "yes", 1, 0)

categorical_cols <- c("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome")
for (col in categorical_cols) {
  data[[col]] <- as.factor(data[[col]])
}

encoded_data <- model.matrix(y ~ . - 1, data = data)
encoded_data <- as.data.frame(encoded_data)
encoded_data$y <- data$y

features_scaled <- as.data.frame(scale(encoded_data[, -ncol(encoded_data)]))
target <- encoded_data$y
data_cls_numeric <- cbind(features_scaled, y = target)
str(data_cls_numeric$y)

data_cls_factor <- data_cls_numeric
data_cls_factor$y <- as.factor(data_cls_factor$y)
levels(data_cls_factor$y) <- c("X0", "X1")

names(data_cls_factor) <- make.names(names(data_cls_factor))

cat("Rozkład klas w danych (wszystkie):\n")
print(table(data_cls_factor$y))

# Definicja zakresu wartości dla hiperparametru k
k_values <- c(3, 5, 7, 9, 11)

results_knn_cls_custom <- data.frame(k = numeric(), accuracy = numeric())

# Iteracja po różnych wartościach k
for (k_val in k_values) {
  hyperparams_knn_cls_custom <- list(k = k_val, threshold = 0.5)
  
  # Przeprowadzenie 5-krotnej CV
  cv_results_knn_cls <- custom_cv(
    data = data_cls_numeric,
    k_folds = 5,
    train_func = train_knn_classification,
    predict_func = predict_knn_classification,
    performance_funcs = list(accuracy = accuracy),
    hyperparams = hyperparams_knn_cls_custom,
    target_col = "y",
    pass_to_predict = TRUE
  )
  
  # Zapisujemy średnią wartość accuracy dla części testowej CV
  results_knn_cls_custom <- rbind(results_knn_cls_custom, 
                                  data.frame(k = k_val,
                                             accuracy = cv_results_knn_cls$test$accuracy$mean))
}

print(results_knn_cls_custom)

# Wybór najlepszego modelu (maksymalna accuracy)
best_knn_model <- results_knn_cls_custom[which.max(results_knn_cls_custom$accuracy), ]
print(best_knn_model)

cat("Najlepsza konfiguracja dla modelu k-NN (klasyfikacja binarna):\n")
cat("Liczba najbliższych sąsiadów (k):", best_knn_model$k, "\n")
cat("Osiągnięta Accuracy:", best_knn_model$accuracy, "\n")

# Wizualizacja wyników – Accuracy w zależności od liczby sąsiadów k
ggplot(results_knn_cls_custom, aes(x = k, y = accuracy)) +
  geom_line() +
  geom_point() +
  ggtitle("Własny model k-NN (klasyfikacja binarna): Accuracy vs. k") +
  xlab("Liczba najbliższych sąsiadów (k)") +
  ylab("Accuracy")


# Definicja zakresu wartości dla hiperparametru depth
depth_values <- c(3, 5, 7, 9, 11)

results_tree_cls_custom <- data.frame(depth = numeric(), accuracy = numeric())

# Iteracja po różnych wartościach depth
for (d in depth_values) {
  hyperparams_tree_cls_custom <- list(depth = d)
  
  # Przeprowadzenie 5-krotnej CV
  cv_results_tree_cls <- custom_cv(
    data = data_cls_numeric,
    k_folds = 5,
    train_func = train_tree_classification,
    predict_func = predict_tree_classification,
    performance_funcs = list(accuracy = accuracy),
    hyperparams = hyperparams_tree_cls_custom,
    target_col = "y",
    pass_to_predict = FALSE
  )
  
  # Zapisujemy średnią wartość accuracy dla części testowej CV
  results_tree_cls_custom <- rbind(results_tree_cls_custom,
                                   data.frame(depth = d,
                                              accuracy = cv_results_tree_cls$test$accuracy$mean))
}

print(results_tree_cls_custom)

# Wybór najlepszego modelu (maksymalna accuracy)
best_tree_model <- results_tree_cls_custom[which.max(results_tree_cls_custom$accuracy), ]
print(best_tree_model)

cat("Najlepsza konfiguracja drzewa decyzyjnego (klasyfikacja binarna):\n")
cat("Maksymalna głębokość:", best_tree_model$depth, "\n")
cat("Osiągnięta Accuracy:", best_tree_model$accuracy, "\n")

# Wizualizacja wyników – Accuracy w zależności od głębokości drzewa
ggplot(results_tree_cls_custom, aes(x = depth, y = accuracy)) +
  geom_line(color = "blue") +
  geom_point(color = "blue") +
  ggtitle("Własny model drzewa decyzyjnego (klasyfikacja binarna): Accuracy vs. Depth") +
  xlab("Maksymalna głębokość drzewa (depth)") +
  ylab("Accuracy")

# Definicja siatki hiperparametrów dla sieci neuronowej
nn_grid <- expand.grid(hidden_neurons = c(10, 30, 50, 70, 100),
                       learning_rate = c(0.0005, 0.001, 0.005))

results_nn_cls_custom <- data.frame(hidden_neurons = numeric(),
                                    learning_rate = numeric(),
                                    accuracy = numeric())

# Iteracja po każdej kombinacji hiperparametrów
for (i in 1:nrow(nn_grid)) {
  h_neurons <- nn_grid$hidden_neurons[i]
  lr <- nn_grid$learning_rate[i]
  
  hyperparams_nn_cls_custom <- list(hidden_neurons = h_neurons,
                                    epochs = 2000,
                                    learning_rate = lr)
  
  # Przeprowadzenie 5-krotnej CV
  cv_results_nn_cls <- custom_cv(
    data = data_cls_numeric,
    k_folds = 5,
    train_func = train_nn_classification,
    predict_func = predict_nn_classification,
    performance_funcs = list(accuracy = accuracy),
    hyperparams = hyperparams_nn_cls_custom,
    target_col = "y",
    pass_to_predict = FALSE
  )
  
  results_nn_cls_custom <- rbind(results_nn_cls_custom,
                                 data.frame(hidden_neurons = h_neurons,
                                            learning_rate = lr,
                                            accuracy = cv_results_nn_cls$test$accuracy$mean))
}

print(results_nn_cls_custom)

# Wybór najlepszego modelu (maksymalna accuracy)
best_nn_model <- results_nn_cls_custom[which.max(results_nn_cls_custom$accuracy), ]
print(best_nn_model)

cat("Najlepsza konfiguracja sieci neuronowej (klasyfikacja binarna):\n")
cat("Liczba neuronów:", best_nn_model$hidden_neurons, "\n")
cat("Learning rate:", best_nn_model$learning_rate, "\n")
cat("Accuracy:", best_nn_model$accuracy, "\n")

# Wizualizacja wyników – Accuracy w zależności od liczby neuronów i współczynnika uczenia
ggplot(results_nn_cls_custom, aes(x = factor(hidden_neurons), y = accuracy, fill = factor(learning_rate))) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Własny model sieci neuronowej (klasyfikacja binarna): Accuracy") +
  xlab("Liczba neuronów w warstwie ukrytej") +
  ylab("Accuracy") +
  labs(fill = "Learning Rate")


hyperparams_knn_cls <- list(k = best_knn_model$k, threshold = 0.5)
hyperparams_tree_cls <- list(depth = best_tree_model$depth)
hyperparams_nn_cls   <- list(hidden_neurons = best_nn_model$hidden_neurons, epochs = 2000, learning_rate = best_nn_model$learning_rate)

set.seed(123)

metrics_knn_cls <- custom_cv(
  data = data_cls_numeric,
  k_folds = 5,
  train_func = train_knn_classification,
  predict_func = predict_knn_classification,
  performance_funcs = list(accuracy = accuracy, precision = precision, recall = recall, f1 = f1_score),
  hyperparams = hyperparams_knn_cls,
  target_col = "y",
  pass_to_predict = TRUE
)

metrics_tree_cls <- custom_cv(
  data = data_cls_numeric,
  k_folds = 5,
  train_func = train_tree_classification,
  predict_func = predict_tree_classification,
  performance_funcs = list(accuracy = accuracy, precision = precision, recall = recall, f1 = f1_score),
  hyperparams = hyperparams_tree_cls,
  target_col = "y",
  pass_to_predict = FALSE
)

metrics_nn_cls <- custom_cv(
  data = data_cls_numeric,
  k_folds = 5,
  train_func = train_nn_classification,
  predict_func = predict_nn_classification,
  performance_funcs = list(accuracy = accuracy, precision = precision, recall = recall, f1 = f1_score),
  hyperparams = hyperparams_nn_cls,
  target_col = "y",
  pass_to_predict = FALSE
)

cat("Wyniki CV dla własnych modeli (Klasyfikacja binarna):\n")
cat("----------------------------------------------------\n")

# Wyniki dla modelu k-NN
cat("Model k-NN:\n")
cat("Trening:\n")
cat("  Accuracy:", metrics_knn_cls$train$accuracy$mean, "(sd:", metrics_knn_cls$train$accuracy$sd, ")\n")
cat("  Precision:", metrics_knn_cls$train$precision$mean, "(sd:", metrics_knn_cls$train$precision$sd, ")\n")
cat("  Recall:", metrics_knn_cls$train$recall$mean, "(sd:", metrics_knn_cls$train$recall$sd, ")\n")
cat("  F1 Score:", metrics_knn_cls$train$f1$mean, "(sd:", metrics_knn_cls$train$f1$sd, ")\n\n")
cat("Walidacja (Test):\n")
cat("  Accuracy:", metrics_knn_cls$test$accuracy$mean, "(sd:", metrics_knn_cls$test$accuracy$sd, ")\n")
cat("  Precision:", metrics_knn_cls$test$precision$mean, "(sd:", metrics_knn_cls$test$precision$sd, ")\n")
cat("  Recall:", metrics_knn_cls$test$recall$mean, "(sd:", metrics_knn_cls$test$recall$sd, ")\n")
cat("  F1 Score:", metrics_knn_cls$test$f1$mean, "(sd:", metrics_knn_cls$test$f1$sd, ")\n\n")
cat("Różnica (Test - Trening):\n")
cat("  Accuracy diff:", metrics_knn_cls$diff$accuracy$mean_diff, "\n")
cat("  Precision diff:", metrics_knn_cls$diff$precision$mean_diff, "\n")
cat("  Recall diff:", metrics_knn_cls$diff$recall$mean_diff, "\n")
cat("  F1 diff:", metrics_knn_cls$diff$f1$mean_diff, "\n")
cat("----------------------------------------------------\n\n")

# Wyniki dla modelu drzewa decyzyjnego
cat("Model drzewa decyzyjnego:\n")
cat("Trening:\n")
cat("  Accuracy:", metrics_tree_cls$train$accuracy$mean, "(sd:", metrics_tree_cls$train$accuracy$sd, ")\n")
cat("  Precision:", metrics_tree_cls$train$precision$mean, "(sd:", metrics_tree_cls$train$precision$sd, ")\n")
cat("  Recall:", metrics_tree_cls$train$recall$mean, "(sd:", metrics_tree_cls$train$recall$sd, ")\n")
cat("  F1 Score:", metrics_tree_cls$train$f1$mean, "(sd:", metrics_tree_cls$train$f1$sd, ")\n\n")
cat("Walidacja (Test):\n")
cat("  Accuracy:", metrics_tree_cls$test$accuracy$mean, "(sd:", metrics_tree_cls$test$accuracy$sd, ")\n")
cat("  Precision:", metrics_tree_cls$test$precision$mean, "(sd:", metrics_tree_cls$test$precision$sd, ")\n")
cat("  Recall:", metrics_tree_cls$test$recall$mean, "(sd:", metrics_tree_cls$test$recall$sd, ")\n")
cat("  F1 Score:", metrics_tree_cls$test$f1$mean, "(sd:", metrics_tree_cls$test$f1$sd, ")\n\n")
cat("Różnica (Test - Trening):\n")
cat("  Accuracy diff:", metrics_tree_cls$diff$accuracy$mean_diff, "\n")
cat("  Precision diff:", metrics_tree_cls$diff$precision$mean_diff, "\n")
cat("  Recall diff:", metrics_tree_cls$diff$recall$mean_diff, "\n")
cat("  F1 diff:", metrics_tree_cls$diff$f1$mean_diff, "\n")
cat("----------------------------------------------------\n\n")

# Wyniki dla modelu sieci neuronowej
cat("Model sieci neuronowej:\n")
cat("Trening:\n")
cat("  Accuracy:", metrics_nn_cls$train$accuracy$mean, "(sd:", metrics_nn_cls$train$accuracy$sd, ")\n")
cat("  Precision:", metrics_nn_cls$train$precision$mean, "(sd:", metrics_nn_cls$train$precision$sd, ")\n")
cat("  Recall:", metrics_nn_cls$train$recall$mean, "(sd:", metrics_nn_cls$train$recall$sd, ")\n")
cat("  F1 Score:", metrics_nn_cls$train$f1$mean, "(sd:", metrics_nn_cls$train$f1$sd, ")\n\n")
cat("Walidacja (Test):\n")
cat("  Accuracy:", metrics_nn_cls$test$accuracy$mean, "(sd:", metrics_nn_cls$test$accuracy$sd, ")\n")
cat("  Precision:", metrics_nn_cls$test$precision$mean, "(sd:", metrics_nn_cls$test$precision$sd, ")\n")
cat("  Recall:", metrics_nn_cls$test$recall$mean, "(sd:", metrics_nn_cls$test$recall$sd, ")\n")
cat("  F1 Score:", metrics_nn_cls$test$f1$mean, "(sd:", metrics_nn_cls$test$f1$sd, ")\n\n")
cat("Różnica (Test - Trening):\n")
cat("  Accuracy diff:", metrics_nn_cls$diff$accuracy$mean_diff, "\n")
cat("  Precision diff:", metrics_nn_cls$diff$precision$mean_diff, "\n")
cat("  Recall diff:", metrics_nn_cls$diff$recall$mean_diff, "\n")
cat("  F1 diff:", metrics_nn_cls$diff$f1$mean_diff, "\n")
cat("----------------------------------------------------\n")


set.seed(123)
train_indices <- createDataPartition(data_cls_factor$y, p = 0.8, list = FALSE)
train_data_cls <- data_cls_factor[train_indices, ]
test_data_cls  <- data_cls_factor[-train_indices, ]
cat("Rozkład klas w zbiorze treningowym:\n")
print(table(train_data_cls$y))
cat("Rozkład klas w zbiorze testowym:\n")
print(table(test_data_cls$y))


folds <- createFolds(train_data_cls$y, k = 5)
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, index = folds)
lapply(folds, function(idx) table(train_data_cls$y[idx]))

# Model k-NN przy użyciu caret
knn_model_pkg <- train(y ~ ., 
                       data = data_cls_factor, 
                       method = "knn", 
                       tuneGrid = data.frame(k = hyperparams_knn_cls$k),
                       trControl = ctrl)
knn_predictions_pkg <- predict(knn_model_pkg, newdata = data_cls_factor)
knn_acc_pkg <- mean(knn_predictions_pkg == data_cls_factor$y)
knn_precision_pkg <- precision(data_cls_factor$y, knn_predictions_pkg)
knn_recall_pkg <- recall(data_cls_factor$y, knn_predictions_pkg)
knn_f1_pkg <- f1_score(data_cls_factor$y, knn_predictions_pkg)

# Model drzewa decyzyjnego przy użyciu rpart
tree_model_pkg <- train(y ~ ., 
                        data = data_cls_factor, 
                        method = "rpart",
                        tuneGrid = data.frame(cp = 0.001),
                        trControl = ctrl,
                        control = rpart.control(maxdepth = hyperparams_tree_cls$depth),
                        metric = "Accuracy")
tree_predictions_pkg <- predict(tree_model_pkg, newdata = data_cls_factor)
tree_acc_pkg <- mean(tree_predictions_pkg == data_cls_factor$y)
tree_precision_pkg <- precision(data_cls_factor$y, tree_predictions_pkg)
tree_recall_pkg <- recall(data_cls_factor$y, tree_predictions_pkg)
tree_f1_pkg <- f1_score(data_cls_factor$y, tree_predictions_pkg)


# Model sieci neuronowej przy użyciu nnet
nn_model_pkg <- train(y ~ ., 
                      data = data_cls_factor, 
                      method = "nnet", 
                      tuneGrid = data.frame(size = hyperparams_nn_cls$hidden_neurons, decay = hyperparams_nn_cls$learning_rate),
                      trControl = ctrl,
                      maxit = hyperparams_nn_cls$epochs,
                      linout = FALSE,
                      trace = FALSE,
                      MaxNWts = 10000)
nn_predictions_pkg <- predict(nn_model_pkg, newdata = data_cls_factor)
nn_acc_pkg <- mean(nn_predictions_pkg == data_cls_factor$y)
nn_precision_pkg <- precision(data_cls_factor$y, nn_predictions_pkg)
nn_recall_pkg <- recall(data_cls_factor$y, nn_predictions_pkg)
nn_f1_pkg <- f1_score(data_cls_factor$y, nn_predictions_pkg)

cat("\nWyniki modeli pakietowych (klasyfikacja binarna):\n")
cat("Caret k-NN Accuracy:", knn_acc_pkg, "\n")
cat("Caret Drzewo Accuracy:", tree_acc_pkg, "\n")
cat("Caret Sieć NN Accuracy:", nn_acc_pkg, "\n")

results_comparison_cls <- data.frame(
  Model = rep(c("k-NN", "Drzewo Decyzyjne", "Sieć Neuronowa"), 2),
  Podejście = rep(c("Własne", "Pakietowe"), each = 3),
  Accuracy = c(metrics_knn_cls$test$accuracy$mean, metrics_tree_cls$test$accuracy$mean, metrics_nn_cls$test$accuracy$mean,
               knn_acc_pkg, tree_acc_pkg, nn_acc_pkg),
  Precision = c(metrics_knn_cls$test$precision$mean, metrics_tree_cls$test$precision$mean, metrics_nn_cls$test$precision$mean,
                knn_precision_pkg, tree_precision_pkg, nn_precision_pkg),
  Recall = c(metrics_knn_cls$test$recall$mean, metrics_tree_cls$test$recall$mean, metrics_nn_cls$test$recall$mean,
             knn_recall_pkg, tree_recall_pkg, nn_recall_pkg),
  F1 = c(metrics_knn_cls$test$f1$mean, metrics_tree_cls$test$f1$mean, metrics_nn_cls$test$f1$mean,
         knn_f1_pkg, tree_f1_pkg, nn_f1_pkg)
)

print(results_comparison_cls)

library(reshape2)
results_long <- melt(results_comparison_cls, id.vars = c("Model", "Podejście"), 
                     variable.name = "Miara", value.name = "Wartość")

ggplot(results_long, aes(x = Model, y = Wartość, fill = Podejście)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  facet_wrap(~ Miara, scales = "free_y") +
  ggtitle("Porównanie wyników modeli: Własne vs. Pakietowe (Klasyfikacja binarna)") +
  theme_minimal()

# ============================================================
# Problem Klasyfikacji Wieloklasowej
# ============================================================

source("funkcje.R")

data <- read.csv("winequality-red.csv", sep = ";")

unique(data$quality)

ggplot(data, aes(x = quality)) +
  geom_bar(fill = "steelblue", color = "black") +
  ggtitle("Rozkład zmiennej targetowanej (quality)") +
  xlab("Jakość wina") +
  ylab("Liczba próbek") +
  theme_minimal()

feature_names <- colnames(data)[-ncol(data)] 

feature_plots <- lapply(feature_names, function(feature) {
  ggplot(data, aes_string(x = feature)) +
    geom_histogram(bins = 30, fill = "darkred", color = "black", alpha = 0.7) +
    ggtitle(paste("Histogram cechy:", feature)) +
    xlab(feature) +
    ylab("Liczba próbek") +
    theme_minimal()
})

do.call(grid.arrange, c(feature_plots[1:4], ncol = 2))

feature_boxplots <- lapply(feature_names, function(feature) {
  ggplot(data, aes(x = factor(quality), y = .data[[feature]])) +
    geom_boxplot(fill = "orange", color = "black", alpha = 0.7) +
    ggtitle(paste("Boxplot cechy:", feature, "względem jakości wina")) +
    xlab("Jakość wina") +
    ylab(feature) +
    theme_minimal()
})

do.call(grid.arrange, c(feature_boxplots[1:4], ncol = 2))


target <- data$quality
features <- data[, -ncol(data)]

features_scaled <- as.data.frame(scale(features))

data_multi <- cbind(features_scaled, quality = target)

# Analiza hiperparametrów dla modeli wieloklasowych
knn_k_values <- c(3, 5, 7, 9, 11)
results_knn_multi_custom <- data.frame(k = numeric(), accuracy = numeric())

for (k_val in knn_k_values) {
  hyperparams_knn_multi_custom <- list(k = k_val)
  
  cv_results_knn_multi <- custom_cv(
    data = data_multi,
    k_folds = 5,
    train_func = train_knn_multiclass,
    predict_func = predict_knn_multiclass,
    performance_funcs = list(accuracy = accuracy),
    hyperparams = hyperparams_knn_multi_custom,
    target_col = "quality",
    pass_to_predict = TRUE
  )
  
  results_knn_multi_custom <- rbind(results_knn_multi_custom,
                                    data.frame(k = k_val,
                                               accuracy = cv_results_knn_multi$test$accuracy$mean))
}

print(results_knn_multi_custom)
best_knn_multi <- results_knn_multi_custom[which.max(results_knn_multi_custom$accuracy), ]
cat("Najlepsza konfiguracja k-NN (klasyfikacja wieloklasowa):\n")
cat("  k =", best_knn_multi$k, "\n")
cat("  Accuracy =", best_knn_multi$accuracy, "\n\n")

ggplot(results_knn_multi_custom, aes(x = k, y = accuracy)) +
  geom_line() +
  geom_point() +
  ggtitle("Własny model k-NN (klasyfikacja wieloklasowa): Accuracy vs. k") +
  xlab("Liczba najbliższych sąsiadów (k)") +
  ylab("Accuracy")


depth_values <- c(3, 5, 7, 9, 11)
results_tree_multi_custom <- data.frame(depth = numeric(), accuracy = numeric())

for (d in depth_values) {
  hyperparams_tree_multi_custom <- list(depth = d)
  
  cv_results_tree_multi <- custom_cv(
    data = data_multi,
    k_folds = 5,
    train_func = train_tree_multiclass,
    predict_func = predict_tree_multiclass,
    performance_funcs = list(accuracy = accuracy),
    hyperparams = hyperparams_tree_multi_custom,
    target_col = "quality",
    pass_to_predict = FALSE
  )
  
  results_tree_multi_custom <- rbind(results_tree_multi_custom,
                                     data.frame(depth = d,
                                                accuracy = cv_results_tree_multi$test$accuracy$mean))
}

print(results_tree_multi_custom)
best_tree_multi <- results_tree_multi_custom[which.max(results_tree_multi_custom$accuracy), ]
cat("Najlepsza konfiguracja drzewa decyzyjnego (klasyfikacja wieloklasowa):\n")
cat("  Depth =", best_tree_multi$depth, "\n")
cat("  Accuracy =", best_tree_multi$accuracy, "\n\n")

ggplot(results_tree_multi_custom, aes(x = depth, y = accuracy)) +
  geom_line(color = "blue") +
  geom_point(color = "blue") +
  ggtitle("Własny model drzewa decyzyjnego (klasyfikacja wieloklasowa): Accuracy vs. Depth") +
  xlab("Głębokość drzewa") +
  ylab("Accuracy")


nn_grid_multi <- expand.grid(hidden_neurons = c(50, 75, 100, 125),
                             learning_rate = c(0.0001, 0.0005, 0.001))
results_nn_multi_custom <- data.frame(hidden_neurons = numeric(),
                                      learning_rate = numeric(),
                                      accuracy = numeric())

for (i in 1:nrow(nn_grid_multi)) {
  h_neurons <- nn_grid_multi$hidden_neurons[i]
  lr <- nn_grid_multi$learning_rate[i]
  
  hyperparams_nn_multi_custom <- list(hidden_neurons = h_neurons,
                                      epochs = 2000,
                                      learning_rate = lr)
  
  cv_results_nn_multi <- custom_cv(
    data = data_multi,
    k_folds = 5,
    train_func = train_nn_multiclass,
    predict_func = predict_nn_multiclass,
    performance_funcs = list(accuracy = accuracy),
    hyperparams = hyperparams_nn_multi_custom,
    target_col = "quality",
    pass_to_predict = FALSE
  )
  
  results_nn_multi_custom <- rbind(results_nn_multi_custom,
                                   data.frame(hidden_neurons = h_neurons,
                                              learning_rate = lr,
                                              accuracy = cv_results_nn_multi$test$accuracy$mean))
}

print(results_nn_multi_custom)
best_nn_multi <- results_nn_multi_custom[which.max(results_nn_multi_custom$accuracy), ]
cat("Najlepsza konfiguracja sieci neuronowej (klasyfikacja wieloklasowa):\n")
cat("  Hidden neurons =", best_nn_multi$hidden_neurons, "\n")
cat("  Learning rate =", best_nn_multi$learning_rate, "\n")
cat("  Accuracy =", best_nn_multi$accuracy, "\n\n")

ggplot(results_nn_multi_custom, aes(x = factor(hidden_neurons), y = accuracy, fill = factor(learning_rate))) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Własny model sieci neuronowej (klasyfikacja wieloklasowa): Accuracy") +
  xlab("Liczba neuronów w warstwie ukrytej") +
  ylab("Accuracy") +
  labs(fill = "Learning Rate")


hyperparams_knn_cls <- list(k = best_knn_multi$k)
hyperparams_tree_cls <- list(depth = best_tree_multi$depth)
hyperparams_nn_cls   <- list(hidden_neurons = best_nn_multi$hidden_neurons, epochs = 2000, learning_rate = best_nn_multi$learning_rate)


set.seed(123)

metrics_knn_multi <- custom_cv(
  data = data_multi,
  k_folds = 5,
  train_func = train_knn_multiclass,
  predict_func = predict_knn_multiclass,
  performance_funcs = list(accuracy = accuracy, precision = macro_precision, recall = macro_recall, f1 = macro_f1),
  hyperparams = hyperparams_knn_cls,
  target_col = "quality",
  pass_to_predict = TRUE
)

metrics_tree_multi <- custom_cv(
  data = data_multi,
  k_folds = 5,
  train_func = train_tree_multiclass,
  predict_func = predict_tree_multiclass,
  performance_funcs = list(accuracy = accuracy, precision = macro_precision, recall = macro_recall, f1 = macro_f1),
  hyperparams = hyperparams_tree_cls,
  target_col = "quality",
  pass_to_predict = FALSE
)

metrics_nn_multi <- custom_cv(
  data = data_multi,
  k_folds = 5,
  train_func = train_nn_multiclass,
  predict_func = predict_nn_multiclass,
  performance_funcs = list(accuracy = accuracy, precision = macro_precision, recall = macro_recall, f1 = macro_f1),
  hyperparams = hyperparams_nn_cls,
  target_col = "quality",
  pass_to_predict = FALSE
)

cat("Wyniki CV dla własnych modeli (Klasyfikacja wieloklasowa):\n")
cat("----------------------------------------------------\n")

# Wyniki dla modelu k-NN
cat("Model k-NN:\n")
cat("Trening:\n")
cat("  Accuracy:", metrics_knn_multi$train$accuracy$mean, "(sd:", metrics_knn_multi$train$accuracy$sd, ")\n")
cat("  Precision:", metrics_knn_multi$train$precision$mean, "(sd:", metrics_knn_multi$train$precision$sd, ")\n")
cat("  Recall:", metrics_knn_multi$train$recall$mean, "(sd:", metrics_knn_multi$train$recall$sd, ")\n")
cat("  F1 Score:", metrics_knn_multi$train$f1$mean, "(sd:", metrics_knn_multi$train$f1$sd, ")\n\n")
cat("Walidacja (Test):\n")
cat("  Accuracy:", metrics_knn_multi$test$accuracy$mean, "(sd:", metrics_knn_multi$test$accuracy$sd, ")\n")
cat("  Precision:", metrics_knn_multi$test$precision$mean, "(sd:", metrics_knn_multi$test$precision$sd, ")\n")
cat("  Recall:", metrics_knn_multi$test$recall$mean, "(sd:", metrics_knn_multi$test$recall$sd, ")\n")
cat("  F1 Score:", metrics_knn_multi$test$f1$mean, "(sd:", metrics_knn_multi$test$f1$sd, ")\n")
cat("Różnica (Test - Trening):\n")
cat("  Accuracy diff:", metrics_knn_multi$diff$accuracy$mean_diff, "\n")
cat("  Precision diff:", metrics_knn_multi$diff$precision$mean_diff, "\n")
cat("  Recall diff:", metrics_knn_multi$diff$recall$mean_diff, "\n")
cat("  F1 diff:", metrics_knn_multi$diff$f1$mean_diff, "\n")
cat("----------------------------------------------------\n\n")

# Wyniki dla modelu drzewa decyzyjnego
cat("Model drzewa decyzyjnego:\n")
cat("Trening:\n")
cat("  Accuracy:", metrics_tree_multi$train$accuracy$mean, "(sd:", metrics_tree_multi$train$accuracy$sd, ")\n")
cat("  Precision:", metrics_tree_multi$train$precision$mean, "(sd:", metrics_tree_multi$train$precision$sd, ")\n")
cat("  Recall:", metrics_tree_multi$train$recall$mean, "(sd:", metrics_tree_multi$train$recall$sd, ")\n")
cat("  F1 Score:", metrics_tree_multi$train$f1$mean, "(sd:", metrics_tree_multi$train$f1$sd, ")\n\n")
cat("Walidacja (Test):\n")
cat("  Accuracy:", metrics_tree_multi$test$accuracy$mean, "(sd:", metrics_tree_multi$test$accuracy$sd, ")\n")
cat("  Precision:", metrics_tree_multi$test$precision$mean, "(sd:", metrics_tree_multi$test$precision$sd, ")\n")
cat("  Recall:", metrics_tree_multi$test$recall$mean, "(sd:", metrics_tree_multi$test$recall$sd, ")\n")
cat("  F1 Score:", metrics_tree_multi$test$f1$mean, "(sd:", metrics_tree_multi$test$f1$sd, ")\n\n")
cat("Różnica (Test - Trening):\n")
cat("  Accuracy diff:", metrics_tree_multi$diff$accuracy$mean_diff, "\n")
cat("  Precision diff:", metrics_tree_multi$diff$precision$mean_diff, "\n")
cat("  Recall diff:", metrics_tree_multi$diff$recall$mean_diff, "\n")
cat("  F1 diff:", metrics_tree_multi$diff$f1$mean_diff, "\n")
cat("----------------------------------------------------\n\n")

# Wyniki dla modelu sieci neuronowej
cat("Model sieci neuronowej:\n")
cat("Trening:\n")
cat("  Accuracy:", metrics_nn_multi$train$accuracy$mean, "(sd:", metrics_nn_multi$train$accuracy$sd, ")\n")
cat("  Precision:", metrics_nn_multi$train$precision$mean, "(sd:", metrics_nn_multi$train$precision$sd, ")\n")
cat("  Recall:", metrics_nn_multi$train$recall$mean, "(sd:", metrics_nn_multi$train$recall$sd, ")\n")
cat("  F1 Score:", metrics_nn_multi$train$f1$mean, "(sd:", metrics_nn_multi$train$f1$sd, ")\n\n")
cat("Walidacja (Test):\n")
cat("  Accuracy:", metrics_nn_multi$test$accuracy$mean, "(sd:", metrics_nn_multi$test$accuracy$sd, ")\n")
cat("  Precision:", metrics_nn_multi$test$precision$mean, "(sd:", metrics_nn_multi$test$precision$sd, ")\n")
cat("  Recall:", metrics_nn_multi$test$recall$mean, "(sd:", metrics_nn_multi$test$recall$sd, ")\n")
cat("  F1 Score:", metrics_nn_multi$test$f1$mean, "(sd:", metrics_nn_multi$test$f1$sd, ")\n\n")
cat("Różnica (Test - Trening):\n")
cat("  Accuracy diff:", metrics_nn_multi$diff$accuracy$mean_diff, "\n")
cat("  Precision diff:", metrics_nn_multi$diff$precision$mean_diff, "\n")
cat("  Recall diff:", metrics_nn_multi$diff$recall$mean_diff, "\n")
cat("  F1 diff:", metrics_nn_multi$diff$f1$mean_diff, "\n")
cat("----------------------------------------------------\n")


# Modele pakietowe (wieloklasowe)

train_indices <- createDataPartition(data_multi$quality, p = 0.8, list = FALSE)
train_data_pkg_multi <- cbind(features_scaled[train_indices, ], quality = as.factor(target[train_indices]))
test_data_pkg_multi  <- cbind(features_scaled[-train_indices, ], quality = as.factor(target[-train_indices]))

levels(train_data_pkg_multi$quality) <- make.names(levels(train_data_pkg_multi$quality))
levels(test_data_pkg_multi$quality)  <- make.names(levels(test_data_pkg_multi$quality))

ctrl <- trainControl(method = "cv", number = 5)

# Model KNN przy użyciu caret
knn_model_pkg_multi <- train(quality ~ ., 
                             data = train_data_pkg_multi, 
                             method = "knn", 
                             tuneGrid = data.frame(k = hyperparams_knn_cls$k),
                             trControl = ctrl)
knn_predictions_pkg_multi <- predict(knn_model_pkg_multi, newdata = test_data_pkg_multi)
knn_acc_pkg_multi <- mean(knn_predictions_pkg_multi == test_data_pkg_multi$quality)
knn_precision_pkg <- macro_precision(test_data_pkg_multi$quality, knn_predictions_pkg_multi)
knn_recall_pkg    <- macro_recall(test_data_pkg_multi$quality, knn_predictions_pkg_multi)
knn_f1_pkg        <- macro_f1(test_data_pkg_multi$quality, knn_predictions_pkg_multi)


# Model drzewa decyzyjnego przy użyciu rpart
tree_model_pkg_multi <- train(quality ~ ., 
                              data = train_data_pkg_multi, 
                              method = "rpart", 
                              tuneGrid = data.frame(cp = 0.01),  # wartość można dostroić
                              trControl = ctrl,
                              control = rpart.control(maxdepth = hyperparams_tree_cls$depth),
                              metric = "Accuracy")
tree_predictions_pkg_multi <- predict(tree_model_pkg_multi, newdata = test_data_pkg_multi)
tree_acc_pkg_multi <- mean(tree_predictions_pkg_multi == test_data_pkg_multi$quality)
tree_precision_pkg <- macro_precision(test_data_pkg_multi$quality, tree_predictions_pkg_multi)
tree_recall_pkg    <- macro_recall(test_data_pkg_multi$quality, tree_predictions_pkg_multi)
tree_f1_pkg        <- macro_f1(test_data_pkg_multi$quality, tree_predictions_pkg_multi)

# Model sieci neuronowej przy użyciu nnet
nn_model_pkg_multi <- train(quality ~ ., 
                            data = train_data_pkg_multi, 
                            method = "nnet", 
                            tuneGrid = data.frame(size = hyperparams_nn_cls$hidden_neurons, decay = hyperparams_nn_cls$learning_rate),
                            trControl = ctrl,
                            maxit = hyperparams_nn_cls$epochs,
                            linout = FALSE,   # dla klasyfikacji
                            trace = FALSE,
                            MaxNWts = 10000)
nn_predictions_pkg_multi <- predict(nn_model_pkg_multi, newdata = test_data_pkg_multi)
nn_acc_pkg_multi <- mean(nn_predictions_pkg_multi == test_data_pkg_multi$quality)
nn_precision_pkg <- macro_precision(test_data_pkg_multi$quality, nn_predictions_pkg_multi)
nn_recall_pkg    <- macro_recall(test_data_pkg_multi$quality, nn_predictions_pkg_multi)
nn_f1_pkg        <- macro_f1(test_data_pkg_multi$quality, nn_predictions_pkg_multi)


cat("\nWyniki modeli pakietowych (wieloklasowych):\n")
cat("Caret KNN Accuracy:", knn_acc_pkg_multi, "\n")
cat("Caret Drzewo Accuracy:", tree_acc_pkg_multi, "\n")
cat("Caret Sieć NN Accuracy:", nn_acc_pkg_multi, "\n")

results_own <- data.frame(
  Model = c("k-NN", "Drzewo Decyzyjne", "Sieć Neuronowa"),
  Accuracy = c(metrics_knn_multi$test$accuracy$mean,
               metrics_tree_multi$test$accuracy$mean,
               metrics_nn_multi$test$accuracy$mean),
  Precision = c(metrics_knn_multi$test$precision$mean,
                metrics_tree_multi$test$precision$mean,
                metrics_nn_multi$test$precision$mean),
  Recall = c(metrics_knn_multi$test$recall$mean,
             metrics_tree_multi$test$recall$mean,
             metrics_nn_multi$test$recall$mean),
  F1 = c(metrics_knn_multi$test$f1$mean,
         metrics_tree_multi$test$f1$mean,
         metrics_nn_multi$test$f1$mean)
)
results_own$Podejście <- "Własne"

results_pkg <- data.frame(
  Model = c("k-NN", "Drzewo Decyzyjne", "Sieć Neuronowa"),
  Accuracy = c(knn_acc_pkg_multi, tree_acc_pkg_multi, nn_acc_pkg_multi),
  Precision = c(knn_precision_pkg, tree_precision_pkg, nn_precision_pkg),
  Recall = c(knn_recall_pkg, tree_recall_pkg, nn_recall_pkg),
  F1 = c(knn_f1_pkg, tree_f1_pkg, nn_f1_pkg)
)
results_pkg$Podejście <- "Pakietowe"

results_comparison_multiclass <- rbind(results_own, results_pkg)

print(results_comparison_multiclass)

results_long <- melt(results_comparison_multiclass, id.vars = c("Model", "Podejście"), 
                     variable.name = "Miara", value.name = "Wartość")

ggplot(results_long, aes(x = Model, y = Wartość, fill = Podejście)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  facet_wrap(~ Miara, scales = "free_y") +
  ggtitle("Porównanie wyników modeli: Własne vs. Pakietowe (klasyfikacja wieloklasowa)") +
  theme_minimal()
