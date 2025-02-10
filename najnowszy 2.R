source("funkcje 3.R")

# Wczytanie bibliotek
library(caret)
library(rpart)
library(nnet)
library(neuralnet)
library(ggplot2)
setwd("C:\\Users\\korna\\Downloads\\projekt")
# Sprawdzamy, czy katalog został ustawiony poprawnie
getwd()

## Regresja 
# Wczytanie danych
data <- read.csv("abalone.csv", sep = ",", header = TRUE, stringsAsFactors = FALSE)
# Sprawdzenie danych wejściowych
cat("Wymiary danych:", dim(data), "\n")
cat("Nazwy kolumn:\n")
print(names(data))
cat("Pierwsze kilka wierszy danych:\n")
print(head(data))

# Usuwanie wierszy z brakującymi wartościami
data_clean <- na.omit(data)
cat("Wymiary danych po usunięciu braków:", dim(data_clean), "\n")

# Podział na dane treningowe i testowe
set.seed(123)
train_idx <- sample(1:nrow(data_clean), 0.8 * nrow(data_clean))
test_idx <- setdiff(1:nrow(data_clean), train_idx)

train_data <- data_clean[train_idx, ]
test_data <- data_clean[test_idx, ]
# Konwersja kolumn nieliczbowych w zbiorach treningowych i testowych
train_data <- as.data.frame(lapply(train_data, safe_numeric_conversion))
test_data <- as.data.frame(lapply(test_data, safe_numeric_conversion))

# Sprawdzenie, które kolumny zawierają wartości nieliczbowe
non_numeric_cols <- sapply(train_data, function(x) !is.numeric(x))
cat("Kolumny nieliczbowe:\n", names(train_data)[non_numeric_cols], "\n")

# Sprawdzenie, czy po konwersji występują wartości NA
cat("Liczba wartości NA w danych treningowych po konwersji:", sum(is.na(train_data)), "\n")
cat("Liczba wartości NA w danych testowych po konwersji:", sum(is.na(test_data)), "\n")

# Zastąpienie wartości NA zerem
train_data[is.na(train_data)] <- 0
test_data[is.na(test_data)] <- 0

# Przyjmujemy, że ostatnia kolumna to zmienna docelowa,
# więc skalujemy tylko zmienne objaśniające, a zmienną docelową pozostawiamy w oryginalnej skali.
target_column <- ncol(train_data)

# Rozdzielenie danych na cechy (X) oraz zmienną docelową (y)
x_train <- train_data[, -target_column]
y_train <- train_data[, target_column]

x_test <- test_data[, -target_column]
y_test <- test_data[, target_column]

# Normalizacja tylko zmiennych objaśniających
x_train_scaled <- scale(x_train)
x_test_scaled <- scale(x_test, 
                       center = attr(x_train_scaled, "scaled:center"), 
                       scale = attr(x_train_scaled, "scaled:scale"))

set.seed(123)
folds_reg <- createFolds(y_train, k = 5)

# CV dla własnego modelu KNN (regresja)
custom_knn_cv <- function(k_value) {
  mse_fold <- numeric(length(folds_reg))
  for(i in seq_along(folds_reg)) {
    fold_idx <- folds_reg[[i]]
    x_tr_cv <- x_train_scaled[-fold_idx, ]
    y_tr_cv <- y_train[-fold_idx]
    x_val_cv <- x_train_scaled[fold_idx, ]
    y_val_cv <- y_train[fold_idx]
    
    preds_cv <- knn_my_regression(x_tr_cv, y_tr_cv, x_val_cv, k = k_value)
    mse_fold[i] <- mse(preds_cv, y_val_cv)
  }
  return(mean(mse_fold))
}

k_values_cv <- 1:10
cv_mse_knn <- sapply(k_values_cv, custom_knn_cv)
best_k_cv <- k_values_cv[which.min(cv_mse_knn)]
cat("CV: Najlepsze k dla własnego KNN (regresja) =", best_k_cv, "ze średnim MSE =", min(cv_mse_knn), "\n")

# CV dla własnego drzewa decyzyjnego (regresja)
custom_tree_cv <- function(depth_value) {
  mse_fold <- numeric(length(folds_reg))
  for(i in seq_along(folds_reg)) {
    fold_idx <- folds_reg[[i]]
    x_tr_cv <- x_train_scaled[-fold_idx, ]
    y_tr_cv <- y_train[-fold_idx]
    x_val_cv <- x_train_scaled[fold_idx, ]
    y_val_cv <- y_train[fold_idx]
    
    tree_mod_cv <- decision_tree_reg(x_tr_cv, y_tr_cv, depth = depth_value)
    preds_cv <- predict_tree_reg(tree_mod_cv, x_val_cv)
    mse_fold[i] <- mse(preds_cv, y_val_cv)
  }
  return(mean(mse_fold))
}

depth_values_cv <- 1:10
cv_mse_tree <- sapply(depth_values_cv, custom_tree_cv)
best_depth_cv <- depth_values_cv[which.min(cv_mse_tree)]
cat("CV: Najlepsza głębokość dla własnego drzewa (regresja) =", best_depth_cv, "ze średnim MSE =", min(cv_mse_tree), "\n")

# CV dla własnej sieci neuronowej (regresja)
custom_nn_cv <- function(hidden_neurons, epochs, lr) {
  mse_fold <- numeric(length(folds_reg))
  for(i in seq_along(folds_reg)) {
    fold_idx <- folds_reg[[i]]
    x_tr_cv <- x_train_scaled[-fold_idx, ]
    y_tr_cv <- y_train[-fold_idx]
    x_val_cv <- x_train_scaled[fold_idx, ]
    y_val_cv <- y_train[fold_idx]
    
    nn_mod_cv <- neural_network_my(x_tr_cv, y_tr_cv, hidden_layers = hidden_neurons, epochs = epochs, learning_rate = lr)
    preds_cv <- predict_nn_reg(nn_mod_cv, x_val_cv)
    mse_fold[i] <- mse(as.numeric(preds_cv), y_val_cv)
  }
  return(mean(mse_fold))
}

nn_cv_mse <- custom_nn_cv(hidden_neurons = c(5), epochs = 100, lr = 0.01)
cat("CV: Własna sieć NN (regresja) dla 5 neuronów, 100 epok, lr=0.01 ma średnie MSE =", nn_cv_mse, "\n")


####KNN
knn_predictions <- knn_my_regression(x_train_scaled, y_train, 
                                     x_test_scaled, k = 3)
cat("Pierwsze wyniki KNN(regresja) :\n", knn_predictions[1:5], "\n")

# Obliczanie dokładności własnego modelu 
mse_knn <- mse(knn_predictions, y_test)
mae_knn <- mean(abs(knn_predictions - y_test))  # Średni błąd bezwzględny 
rmse_knn <- sqrt(mean((knn_predictions - y_test)^2))  # Pierwiastek błędu kwadratowego
r2_knn <- 1 - sum((knn_predictions - y_test)^2) / sum((y_test - mean(y_test))^2)  # Współczynnik determinacji R^2

cat("MSE dla własnego KNN (regresja):", mse_knn, "\n")
cat("MAE dla własnego KNN (regresja):", mae_knn, "\n")
cat("RMSE dla własnego KNN (regresja):", rmse_knn, "\n")
cat("R² dla własnego KNN(regresja):", r2_knn, "\n")

#Anliza hiperparametrów
wyniki <- data.frame(k = numeric(), 
                     mse = numeric(), 
                     mae = numeric(), 
                     rmse = numeric(), 
                     r2 = numeric())

# Pętla po wartościach k od 1 do 10
for (k in 1:10) {
  # Obliczenie predykcji dla bieżącej wartości k
  knn_predictions <- knn_my_regression(x_train_scaled, y_train, x_test_scaled, k = k)
  
  
  # Obliczanie metryk jakości modelu
  mse_knn <- mse(knn_predictions, y_test)
  mae_knn <- mean(abs(knn_predictions - y_test))           
  rmse_knn <- sqrt(mean((knn_predictions - y_test)^2))       
  r2_knn <- 1 - sum((knn_predictions - y_test)^2) / sum((y_test - mean(y_test))^2)  
  
  # Dodanie wyników dla aktualnego k do ramki danych
  wyniki <- rbind(wyniki, data.frame(k = k, mse = mse_knn, mae = mae_knn, rmse = rmse_knn, r2 = r2_knn))
}

# Wyświetlenie wyników
print(wyniki)
# Wybór najlepszego k na podstawie minimalnego MSE
best_k_regr <- wyniki$k[which.min(wyniki$mse)]
cat("Najlepsze k (minimalne MSE):", best_k_regr, "\n")

#wbudowane knn
knn_model <- train(x_train_scaled, y_train, method = "knn", 
                   tuneGrid = data.frame(k = 3))
knn_model_predictions <- predict(knn_model, newdata = x_test_scaled)
cat("Liczba predykcji w wbudowanym modelu:", length(knn_model_predictions), "\n")

# Metryki dla wbudowanego modelu
mse_knn_model <- mse(knn_model_predictions, y_test)
mae_knn_model <- mean(abs(knn_model_predictions - y_test))
rmse_knn_model <- sqrt(mean((knn_model_predictions - y_test)^2))
r2_knn_model <- 1 - sum((knn_model_predictions - y_test)^2) / sum((y_test - mean(y_test))^2)

cat("MSE dla wbudowanego KNN(regresja):", mse_knn_model, "\n")
cat("MAE dla wbudowanego KNN (regresja):", mae_knn_model, "\n")
cat("RMSE dla wbudowanego KNN (regresja):", rmse_knn_model, "\n")
cat("R² dla wbudowanego KNN(regresja):", r2_knn_model, "\n")

metrics <- data.frame(
  Model = rep(c("KNN", "KNN_wbudowany"), each = 3),
  Metric = rep(c("MSE", "MAE", "RMSE"), 2),
  Value = c(mse_knn, mae_knn, rmse_knn, mse_knn_model, mae_knn_model, rmse_knn_model)
)

# Rysujemy wykres słupkowy porównujący metryki KNN
ggplot(metrics, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Porównanie wyników błędu KNN: Własny vs Wbudowany", y = "Wartość metryki", x = "Metryka") +
  theme_minimal()


# Wywołanie własnej funkcji drzewa decyzyjnego (maksymalna głębokość = 5)
tree_decision <- decision_tree_reg(x_train_scaled, y_train, depth = 5)
tree_predictions <- predict_tree_reg(tree_decision, x_test_scaled)
cat("Pierwsze wyniki drzewa decyzyjnego (Własne):\n", tree_predictions[1:5], "\n")

# Obliczanie metryk dla własnego drzewa decyzyjnego
mse_tree <- mse(tree_predictions, y_test)
mae_tree <- mean(abs(tree_predictions - y_test))
rmse_tree <- sqrt(mean((tree_predictions - y_test)^2))
r2_tree <- 1 - sum((tree_predictions - y_test)^2) / sum((y_test - mean(y_test))^2)

cat("MSE dla własnego drzewa:", mse_tree, "\n")
cat("MAE dla własnego drzewa:", mae_tree, "\n")
cat("RMSE dla własnego drzewa:", rmse_tree, "\n")
cat("R² dla własnego drzewa:", r2_tree, "\n")

# Przygotowanie danych treningowych – łączenie cech (x_train_scaled) oraz zmiennej docelowej (y_train_scaled)
train_data_dt <- as.data.frame(x_train_scaled)
train_data_dt$target <- y_train   # nazwa kolumny celu to "target"

# Przygotowanie danych testowych – zbiór x_test_scaled (przekształcony do data.frame)
test_data_dt <- as.data.frame(x_test_scaled)
# Trenowanie drzewa decyzyjnego (wbudowanego) z maksymalną głębokością równą 5
tree_model <- train(target ~ ., 
                    data = train_data_dt, 
                    method = "rpart",
                    tuneGrid = data.frame(cp = 0.01),   
                    trControl = trainControl(method = "none"),
                    control = rpart.control(maxdepth = 5))

tree_model_predictions <- predict(tree_model, newdata = test_data_dt)
cat("Pierwsze wyniki drzewa decyzyjnego (Wbudowane):\n", tree_model_predictions[1:5], "\n")

# Obliczanie metryk dla wbudowanego drzewa decyzyjnego
mse_tree_model <- mse(tree_model_predictions, y_test)
mae_tree_model <- mean(abs(tree_model_predictions - y_test))
rmse_tree_model <- sqrt(mean((tree_model_predictions - y_test)^2))
r2_tree_model <- 1 - sum((tree_model_predictions - y_test)^2) / sum((y_test - mean(y_test))^2)

cat("MSE dla wbudowanego drzewa:", mse_tree_model, "\n")
cat("MAE dla wbudowanego drzewa:", mae_tree_model, "\n")
cat("RMSE dla wbudowanego drzewa:", rmse_tree_model, "\n")
cat("R² dla wbudowanego drzewa:", r2_tree_model, "\n")

#wykres słupkowy porównujący metryki TREE
metrics_tree <- data.frame(
  Model = rep(c("Drzewo_własne", "Drzewo_wbudowane"), each = 3),
  Metric = rep(c("MSE", "MAE", "RMSE"), 2),
  Value = c(mse_tree, mae_tree, rmse_tree, mse_tree_model, mae_tree_model, rmse_tree_model)
)

ggplot(metrics_tree, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Porównanie wyników błędu drzewa decyzyjnego:\nWłasne vs Wbudowane", 
       y = "Wartość metryki", 
       x = "Metryka") +
  theme_minimal()

# Sieć neuronowa własna 
x_train_scaled <- as.matrix(x_train_scaled)
x_test_scaled  <- as.matrix(x_test_scaled)
x_test_scaled <- cbind(1, x_test_scaled)

# Trening sieci neuronowej – uwaga: wagi są przechowywane w liście model$W, więc funkcja predict_nn_reg odwołuje się do model$W[[l]]
nn_model_regression <- neural_network_my(x_train_scaled, y_train, hidden_layers = 5, epochs = 100, learning_rate = 0.01)

# Predykcja – wywołanie funkcji predict_nn_reg, która korzysta z poprawnych odwołań do wag
nn_predictions_regression <- as.numeric(predict_nn_reg(nn_model_regression, x_test_scaled))
cat("Pierwsze wyniki sieci neuronowej (Własna):\n", nn_predictions_regression[1:5], "\n")


# Obliczanie metryk dla własnej sieci neuronowej
mse_nn <- mse(nn_predictions_regression, y_test)
mae_nn <- mean(abs(nn_predictions_regression - y_test))
rmse_nn <- sqrt(mean((nn_predictions_regression - y_test)^2))
r2_nn <- 1 - sum((nn_predictions_regression - y_test)^2) / 
  sum((y_test - mean(y_test))^2)

cat("Własna sieć neuronowa:\n")
cat("  MSE:", mse_nn, "\n")
cat("  MAE:", mae_nn, "\n")
cat("  RMSE:", rmse_nn, "\n")
cat("  R²:", r2_nn, "\n\n")

#Wbudowane sieci neuronowe
nn_model_builtin <- nnet(x_train_scaled, y_train, 
                         size = 5, linout = TRUE, maxit = 100, decay = 0.01, trace = FALSE)

# Predykcja wbudowanej sieci neuronowej
nn_predictions_builtin <- predict(nn_model_builtin, x_test_scaled)

# Obliczanie metryk dla wbudowanej sieci neuronowej
mse_nn_builtin <- mse(nn_predictions_builtin, y_test)
mae_nn_builtin <- mean(abs(nn_predictions_builtin - y_test))
rmse_nn_builtin <- sqrt(mean((nn_predictions_builtin - y_test)^2))
r2_nn_builtin <- 1 - sum((nn_predictions_builtin - y_test)^2) / 
  sum((y_test - mean(y_test))^2)

cat("Wbudowana sieć neuronowa:\n")
cat("  MSE:", mse_nn_builtin, "\n")
cat("  MAE:", mae_nn_builtin, "\n")
cat("  RMSE:", rmse_nn_builtin, "\n")
cat("  R²:", r2_nn_builtin, "\n")

#Porównanie własnych sieci i wbudowanych.
metrics_nn <- data.frame(
  Model = rep(c("Własna_NN", "Wbudowana_NN"), each = 3),
  Metric = rep(c("MSE", "MAE", "RMSE"), 2),
  Value = c(mse_nn, mae_nn, rmse_nn, mse_nn_builtin, mae_nn_builtin, rmse_nn_builtin)
)

ggplot(metrics_nn, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Porównanie wyników błędu sieci neuronowych:\nWłasna vs Wbudowana", 
       y = "Wartość metryki", 
       x = "Metryka") +
  theme_minimal()

# Porównanie zgodności własnych funkcji z modelami wbudowanymi

# Dla KNN:
diff_knn <- abs(mse_knn - mse_knn_model) +
  abs(mae_knn - mae_knn_model) +
  abs(rmse_knn - rmse_knn_model) +
  abs(r2_knn - r2_knn_model)

# Dla drzewa decyzyjnego:
diff_tree <- abs(mse_tree - mse_tree_model) +
  abs(mae_tree - mae_tree_model) +
  abs(rmse_tree - rmse_tree_model) +
  abs(r2_tree - r2_tree_model)

# Dla sieci neuronowych:
diff_nn <- abs(mse_nn - mse_nn_builtin) +
  abs(mae_nn - mae_nn_builtin) +
  abs(rmse_nn - rmse_nn_builtin) +
  abs(r2_nn - r2_nn_builtin)

# Sumaryczne różnice zapisujemy w wektorze:
diffs <- c(KNN = diff_knn, Drzewo = diff_tree, NN = diff_nn)

cat("\nSuma różnic między metrykami własnych funkcji a modelami wbudowanymi:\n")
print(diffs)

# Wybieramy metodę, dla której suma różnic jest najmniejsza
best_model <- names(which.min(diffs))
cat("\nNajwiększą zgodność z modelem wbudowanym wykazuje metoda:", best_model, "\n")

# Klasyfikacja binarna 
###################################

source("funkcje.R")
# Wczytanie danych
data_b <- read.csv("winequality-white.csv", sep = ";")

# Podstawowa analiza danych
cat("Pierwsze wiersze danych:\n")
print(head(data_b))
cat("Wymiary danych:", dim(data_b), "\n")
cat("Kolumny danych:\n")
print(names(data_b))

# Sprawdzenie brakujących wartości
missing_values <- colSums(is.na(data_b))
cat("Brakujące wartości w kolumnach:\n")
print(missing_values)

# Wizualizacja rozkładu jakości przed przekształceniem
ggplot(data_b, aes(x = factor(quality))) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Rozkład jakości win przed przekształceniem", x = "Jakość", y = "Liczba win") +
  theme_minimal()

# Przekształcenie kolumny 'quality' na klasy binarne
data_b$quality <- ifelse(data_b$quality >= 6, 1, 0)

# Podział na dane treningowe i testowe
set.seed(123)
train_idx <- sample(1:nrow(data_b), 0.8 * nrow(data_b))
train_data <- data_b[train_idx, -which(names(data_b) == "quality")]
train_labels <- data_b$quality[train_idx]
test_data <- data_b[-train_idx, -which(names(data_b) == "quality")]
test_labels <- data_b$quality[-train_idx]

test_data <- as.matrix(test_data)

cat("Liczba danych treningowych:", length(train_idx), "\n")
cat("Liczba danych testowych:", nrow(data_b) - length(train_idx), "\n")

# Normalizacja
train_data <- scale(train_data)
test_data <- scale(test_data, center = attr(train_data, "scaled:center"), scale = attr(train_data, "scaled:scale"))

set.seed(123)
folds_class <- createFolds(train_labels, k = 5)

# CV dla własnego modelu KNN (klasyfikacja)
custom_knn_class_cv <- function(k_value) {
  acc_fold <- numeric(length(folds_class))
  for(i in seq_along(folds_class)) {
    fold_idx <- folds_class[[i]]
    x_tr_cv <- train_data[-fold_idx, ]
    y_tr_cv <- train_labels[-fold_idx]
    x_val_cv <- train_data[fold_idx, ]
    y_val_cv <- train_labels[fold_idx]
    
    preds_cv <- knn_binary(x_tr_cv, y_tr_cv, x_val_cv, k = k_value)
    acc_fold[i] <- accuracy_b(preds_cv, y_val_cv)
  }
  return(mean(acc_fold))
}

k_values_class_cv <- seq(1, 20, by = 2)
cv_acc_knn_class <- sapply(k_values_class_cv, custom_knn_class_cv)
best_k_class_cv <- k_values_class_cv[which.max(cv_acc_knn_class)]
cat("CV: Najlepsze k dla własnego KNN (klasyfikacja) =", best_k_class_cv, "ze średnią accuracy =", max(cv_acc_knn_class), "\n")

# CV dla własnego drzewa decyzyjnego (klasyfikacja)
custom_tree_class_cv <- function(depth_value) {
  acc_fold <- numeric(length(folds_class))
  for(i in seq_along(folds_class)) {
    fold_idx <- folds_class[[i]]
    x_tr_cv <- train_data[-fold_idx, ]
    y_tr_cv <- train_labels[-fold_idx]
    x_val_cv <- train_data[fold_idx, ]
    y_val_cv <- train_labels[fold_idx]
    
    tree_mod_cv <- decision_tree_binary(x_tr_cv, y_tr_cv, depth = depth_value)
    preds_cv <- predict_tree_binary(tree_mod_cv, x_val_cv)
    acc_fold[i] <- accuracy_b(preds_cv, y_val_cv)
  }
  return(mean(acc_fold))
}

depth_values_class_cv <- 1:10
cv_acc_tree_class <- sapply(depth_values_class_cv, custom_tree_class_cv)
best_depth_class_cv <- depth_values_class_cv[which.max(cv_acc_tree_class)]
cat("CV: Najlepsza głębokość dla własnego drzewa (klasyfikacja) =", best_depth_class_cv, "ze średnią accuracy =", max(cv_acc_tree_class), "\n")

# CV dla własnej sieci neuronowej (klasyfikacja)
custom_nn_class_cv <- function(hidden_neurons, epochs, lr) {
  acc_fold <- numeric(length(folds_class))
  for(i in seq_along(folds_class)) {
    fold_idx <- folds_class[[i]]
    x_tr_cv <- train_data[-fold_idx, ]
    y_tr_cv <- train_labels[-fold_idx]
    x_val_cv <- train_data[fold_idx, ]
    y_val_cv <- train_labels[fold_idx]
    
    nn_mod_cv <- neural_network_binary(x_tr_cv, y_tr_cv, hidden_neurons = hidden_neurons, epochs = epochs, learning_rate = lr)
    preds_cv <- predict_nn_binary(nn_mod_cv, x_val_cv)
    acc_fold[i] <- accuracy_b(preds_cv, y_val_cv)
  }
  return(mean(acc_fold))
}

nn_class_cv_acc <- custom_nn_class_cv(hidden_neurons = 5, epochs = 100, lr = 0.01)
cat("CV: Własna sieć NN (klasyfikacja) dla 5 neuronów, 100 epok, lr=0.01 ma średnią accuracy =", nn_class_cv_acc, "\n")


#Własna implementacja KNN
knn_predictions <- knn_binary(train_data, train_labels, test_data, k = 3)
knn_accuracy <- accuracy_b(knn_predictions, test_labels)
cat("Dokładność KNN dla k=3 (własne):", knn_accuracy, "\n")
#Analiza dokładności modelu w zależności od wartości k
k_values <- seq(1, 20, by = 2)
accuracies <- numeric(length(k_values))

# Pętla przez wybrane wartości k
for(i in seq_along(k_values)) {
  current_k <- k_values[i]
  # Obliczamy predykcje dla aktualnego k
  knn_pred <- knn_binary(train_data, train_labels, test_data, k = current_k)
  # Obliczamy dokładność modelu
  accuracies[i] <- accuracy_b(knn_pred, test_labels)
  cat("Dla k =", current_k, "dokładność:", accuracies[i], "\n")
}

results <- data.frame(k = k_values, accuracy = accuracies)

# Wybieramy najlepszą wartość k (pod względem max. dokładność)
best_index <- which.max(results$accuracy)
best_k <- results$k[best_index]
best_accuracy <- results$accuracy[best_index]
cat("Najlepszy wynik uzyskano dla k =", best_k, "z dokładnością:", best_accuracy, "\n")

# Wizualizacja wyników
ggplot(results, aes(x = k, y = accuracy)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  # Zaznaczamy najlepszą wartość k
  geom_point(data = results[best_index, ], aes(x = k, y = accuracy), color = "green", size = 4) +
  labs(title = "Wpływ parametru k na dokładność modelu KNN",
       x = "Liczba sąsiadów (k)",
       y = "Dokładność") +
  theme_minimal()

knn_predictions <- knn_binary(train_data, train_labels, test_data, k = 13)
knn_accuracy <- accuracy_b(knn_predictions, test_labels)
cat("Dokładność KNN dla k=13 (własne):", knn_accuracy, "\n")
#Wbudowany model KNN 

train_labels <- as.factor(train_labels)
test_labels <- as.factor(test_labels)

# Trening wbudowanego KNN
knn_model <- train(train_data, train_labels, method = "knn", tuneGrid = data.frame(k = 13))
knn_model_predictions <- predict(knn_model, test_data)

# Obliczenie dokładności
knn_model_accuracy <- accuracy_b(knn_model_predictions, test_labels)
cat("Dokładność KNN (wbudowane):", knn_model_accuracy, "\n")

#PORÓWNANIE MODELI KNN DLA NKLASYFIKACJI BINARNEJ
cat("dla k=13", "\n")
cm_knn <- confusionMatrix(as.factor(knn_predictions), as.factor(test_labels), positive = "1")
sensitivity_knn <- cm_knn$byClass["Sensitivity"]
specificity_knn <- cm_knn$byClass["Specificity"]
cat("Czułość własnego modelu :", round(sensitivity_knn, 13), "\n")
cat("Specyficzność włąsnego modelu :", round(specificity_knn, 13), "\n")
cat("Dokładność KNN dla własnego :", knn_accuracy, "\n")


cm_knn_model <- confusionMatrix(knn_model_predictions, test_labels, positive = "1")
sensitivity_knn_model <- cm_knn_model$byClass["Sensitivity"]
specificity_knn_model <- cm_knn_model$byClass["Specificity"]
cat("Czułość wbudowane:", round(sensitivity_knn_model, 13), "\n")
cat("Specyficzność wbudowane):", round(specificity_knn_model, 13), "\n")
cat("Dokładność KNN wbudowane :", knn_model_accuracy, "\n")


# Utworzenie ramki danych z wynikami
comparison_df <- data.frame(
  Model = c("Własne knn", "Wbudowane knn"),
  Accuracy = c(knn_accuracy, knn_model_accuracy),
  Sensitivity = c(as.numeric(sensitivity_knn), as.numeric(sensitivity_knn_model)),
  Specificity = c(as.numeric(specificity_knn), as.numeric(specificity_knn_model))
)
comparison_df$CombinedScore <- rowMeans(comparison_df[, c("Accuracy", "Sensitivity", "Specificity")])

# Wyświetlenie tabeli wyników w konsoli
cat("Porównanie wyników:\n")
print(comparison_df)

best_model <- comparison_df$Model[which.max(comparison_df$CombinedScore)]
cat("Lepszy model (łączny wynik):", best_model, "\n")

############################jeszce nie wiem cy to dawać 
# Wizualizacja wyników na wykresie słupkowym

comparison_melt <- melt(comparison_df, id.vars = "Model", variable.name = "Metric", value.name = "Value")

# Rysowanie wykresu

ggplot(comparison_melt, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Porównanie wyników KNN dla klasyfikacji binarnej",
       x = "Metryka",
       y = "Wartość") +
  theme_minimal()
#######################################
#Własne drzewo decyzyjne
tree_model <- decision_tree_binary(train_data, train_labels, depth = 5)
tree_predictions <- predict_tree_binary(tree_model, test_data)
tree_accuracy <- accuracy_b(tree_predictions, test_labels)
cat("Dokładność drzewa decyzyjnego z głębokością 5 (własne):", tree_accuracy, "\n")

#Analiza dokładności w zależności od głębokości 
depth_values <- 1:20
tree_accuracies <- numeric(length(depth_values))

for(i in seq_along(depth_values)) {
  current_depth <- depth_values[i]
  
  tree_model_temp <- decision_tree_binary(train_data, train_labels, depth = current_depth)
  
  tree_pred <- predict_tree_binary(tree_model_temp, test_data)
  
  # Obliczenie dokładności
  tree_accuracies[i] <- accuracy_b(tree_pred, test_labels)
  cat("Głębokość =", current_depth, "dokładność:", tree_accuracies[i], "\n")
}

results_tree <- data.frame(depth = depth_values, accuracy = tree_accuracies)

cat("Ramka danych z wynikami:\n")
print(results_tree)

best_index <- which.max(results_tree$accuracy)
best_depth <- results_tree$depth[best_index]

cat("Najlepsza głębokość drzewa:", best_depth, "\n")

tree_model <- decision_tree_binary(train_data, train_labels, depth = 17)
tree_predictions <- predict_tree_binary(tree_model, test_data)
tree_accuracy <- accuracy_b(tree_predictions, test_labels)
cat("Dokładność drzewa decyzyjnego z głębokością 17 (własne):", tree_accuracy, "\n")

#Wbudowane drzewo decyzyjne
train_data_dt <- as.data.frame(train_data)
train_data_dt$target <- train_labels
test_data_dt <- as.data.frame(test_data)

tree_model_builtin <- train(target ~ ., data = train_data_dt, method = "rpart", tuneGrid = data.frame(cp = 0.01))
tree_model_predictions <- predict(tree_model_builtin, newdata = test_data_dt)
tree_model_accuracy <- accuracy_b(tree_model_predictions, test_labels)
cat("Dokładność drzewa decyzyjnego (wbudowane):", tree_model_accuracy, "\n")

#PORÓWNANIE MODELI DRZEWA DECYZYJNEGO
cat("Dla drzewa decyzyjnego, głębokość = 17", "\n")

# Dla własnego drzewa decyzyjnego (własna implementacja)
cm_tree <- confusionMatrix(as.factor(tree_predictions), as.factor(test_labels), positive = "1")
sensitivity_tree <- cm_tree$byClass["Sensitivity"]
specificity_tree <- cm_tree$byClass["Specificity"]

cat("Czułość własnego drzewa:", round(sensitivity_tree, 3), "\n")
cat("Specyficzność własnego drzewa:", round(specificity_tree, 3), "\n")
cat("Dokładność własnego drzewa decyzyjnego:", tree_accuracy, "\n")

# Dla wbudowanego drzewa decyzyjnego (train rpart)
cm_tree_builtin <- confusionMatrix(tree_model_predictions, test_labels, positive = "1")
sensitivity_tree_builtin <- cm_tree_builtin$byClass["Sensitivity"]
specificity_tree_builtin <- cm_tree_builtin$byClass["Specificity"]

cat("Czułość wbudowanego drzewa:", round(sensitivity_tree_builtin, 3), "\n")
cat("Specyficzność wbudowanego drzewa:", round(specificity_tree_builtin, 3), "\n")
cat("Dokładność drzewa decyzyjnego wbudowanego:", tree_model_accuracy, "\n")

# Utworzenie ramki danych z wynikami
comparison_df_tree <- data.frame(
  Model = c("Własne Tree", "Wbudowane Tree"),
  Accuracy = c(as.numeric(tree_accuracy), as.numeric(tree_model_accuracy)),
  Sensitivity = c(as.numeric(sensitivity_tree), as.numeric(sensitivity_tree_builtin)),
  Specificity = c(as.numeric(specificity_tree), as.numeric(specificity_tree_builtin))
)

# Obliczenie łącznego wyniku jako średniej wartości Accuracy, Sensitivity i Specificity
comparison_df_tree$CombinedScore <- rowMeans(comparison_df_tree[, c("Accuracy", "Sensitivity", "Specificity")])

cat("Porównanie wyników drzewa decyzyjnego:\n")
print(comparison_df_tree)

# Określenie, który model jest lepszy (model o najwyższym CombinedScore)
best_model_tree <- comparison_df_tree$Model[which.max(comparison_df_tree$CombinedScore)]
cat("Lepszy model drzewa (łączny wynik):", best_model_tree, "\n")

# Wizualizacja wyników na wykresie słupkowym
library(reshape2)
comparison_tree_melt <- melt(comparison_df_tree, id.vars = "Model", variable.name = "Metric", value.name = "Value")

library(ggplot2)
ggplot(comparison_tree_melt, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Porównanie wyników drzewa decyzyjnego",
       x = "Metryka",
       y = "Wartość") +
  theme_minimal()


#Własna sieć neuronowa
nn_model <- neural_network_binary(train_data, train_labels, hidden_neurons = c(8), epochs = 20, learning_rate = 0.01)
nn_predictions <- predict_nn_binary(nn_model, test_data)
nn_accuracy <- accuracy_b(nn_predictions, test_labels)
cat("Dokładność sieci neuronowej dla hidden_neurons = 8, epochs = 200, learning_rate = 0.01 (własne):", nn_accuracy, "\n")

#Analiza hiperparametrów 
# Zakresu hiperparametrów
hidden_neurons_values <- seq(5, 30, by = 5)        
epochs_values <- seq(50, 300, by = 50)               
learning_rates <- c(0.001, 0.005, 0.01, 0.05, 0.1)   
# Przygotowanie pustej ramki danych do zapisywania wyników
grid_results <- data.frame(hidden_neurons = integer(),
                           epochs = integer(),
                           learning_rate = numeric(),
                           accuracy = numeric(),
                           stringsAsFactors = FALSE)

# Pętla iterująca przez wszystkie kombinacje hiperparametrów
for (h in hidden_neurons_values) {
  for (e in epochs_values) {
    for (lr in learning_rates) {
      
      # Trening modelu z bieżącą kombinacją hiperparametrów
      nn_model_temp <- neural_network_binary(train_data, train_labels, 
                                             hidden_neurons = h, 
                                             epochs = e, 
                                             learning_rate = lr)
      
      # Predykcja na zbiorze testowym
      nn_pred <- predict_nn_binary(nn_model_temp, test_data)
      
      # Obliczenie dokładności
      acc <- accuracy_b(nn_pred, test_labels)
      
      # Dodanie wyników do ramki danych
      grid_results <- rbind(grid_results,
                            data.frame(hidden_neurons = h,
                                       epochs = e,
                                       learning_rate = lr,
                                       accuracy = acc))
      
      # Informacja o postępie w konsoli
      cat("hidden_neurons =", h, "epochs =", e, "learning_rate =", lr, "accuracy =", acc, "\n")
    }
  }
}

# Wyświetlenie ramki danych z wynikami
cat("Wyniki Grid Search:\n")
print(grid_results)

# Wybór najlepszej kombinacji (najwyższa dokładność)
best_index <- which.max(grid_results$accuracy)
best_params <- grid_results[best_index, ]
cat("Najlepsza konfiguracja hiperparametrów:\n")
print(best_params)

#Własna sieć neuronowa
nn_model <- neural_network_binary(train_data, train_labels, hidden_neurons =c(5), epochs = 50, learning_rate = 0.001)
nn_predictions <- predict_nn_binary(nn_model, test_data)
nn_accuracy <- accuracy_b(nn_predictions, test_labels)
cat("Dokładność sieci neuronowej dla hidden_neurons = 5, epochs = 50, learning_rate = 0.001 (własne):", nn_accuracy, "\n")

# Wbudowana sieć neuronowa 

train_labels_onehot <- class.ind(train_labels)
test_labels_onehot <- class.ind(test_labels)

# Trening wbudowanej sieci neuronowej
nn_model_builtin <- nnet(train_data, train_labels_onehot, size = 10, maxit = 200, trace = FALSE)
nn_model_builtin_predictions <- predict(nn_model_builtin, test_data, type = "raw")

# Konwersja predykcji do etykiet klasowych (wybór klasy z największym prawdopodobieństwem)
nn_model_builtin_predictions_labels <- ifelse(nn_model_builtin_predictions[, 2] > 0.5, 1, 0)

# Obliczenie dokładności
nn_model_builtin_accuracy <- accuracy_b(nn_model_builtin_predictions_labels, test_labels)
cat("Dokładność wbudowanej sieci neuronowej:", nn_model_builtin_accuracy, "\n")


#PORÓWNANIE MODELI SIECI NEURONOWYCH 
cat("Dla sieci neuronowych:\n")

# Własna sieć neuronowa
cm_nn <- confusionMatrix(as.factor(nn_predictions), as.factor(test_labels), positive = "1")
sensitivity_nn <- cm_nn$byClass["Sensitivity"]
specificity_nn <- cm_nn$byClass["Specificity"]
accuracy_nn <- cm_nn$overall["Accuracy"]

cat("Czułość własnej sieci:", round(sensitivity_nn, 13), "\n")
cat("Specyficzność własnej sieci:", round(specificity_nn, 13), "\n")
cat("Dokładność własnej sieci:", round(accuracy_nn, 13), "\n")

# Wbudowana sieć neuronowa
cm_nn_builtin <- confusionMatrix(as.factor(nn_model_builtin_predictions_labels), as.factor(test_labels), positive = "1")
sensitivity_nn_builtin <- cm_nn_builtin$byClass["Sensitivity"]
specificity_nn_builtin <- cm_nn_builtin$byClass["Specificity"]
accuracy_nn_builtin <- cm_nn_builtin$overall["Accuracy"]

cat("Czułość wbudowanej sieci:", round(sensitivity_nn_builtin, 13), "\n")
cat("Specyficzność wbudowanej sieci:", round(specificity_nn_builtin, 13), "\n")
cat("Dokładność wbudowanej sieci:", round(accuracy_nn_builtin, 13), "\n")

# Utworzenie ramki danych z wynikami
comparison_df_nn <- data.frame(
  Model = c("Własna NN", "Wbudowana NN"),
  Accuracy = c(as.numeric(accuracy_nn), as.numeric(accuracy_nn_builtin)),
  Sensitivity = c(as.numeric(sensitivity_nn), as.numeric(sensitivity_nn_builtin)),
  Specificity = c(as.numeric(specificity_nn), as.numeric(specificity_nn_builtin))
)

# Obliczenie łącznego wyniku jako średniej wartości Accuracy, Sensitivity oraz Specificity
comparison_df_nn$CombinedScore <- rowMeans(comparison_df_nn[, c("Accuracy", "Sensitivity", "Specificity")])

cat("Porównanie wyników sieci neuronowych:\n")
print(comparison_df_nn)

# Określenie, który model jest lepszy (model o najwyższym CombinedScore)
best_model_nn <- comparison_df_nn$Model[which.max(comparison_df_nn$CombinedScore)]
cat("Lepszy model (łączny wynik sieci neuronowych):", best_model_nn, "\n")

# Wizualizacja wyników na wykresie słupkowym
library(reshape2)
comparison_nn_melt <- melt(comparison_df_nn, id.vars = "Model", variable.name = "Metric", value.name = "Value")

library(ggplot2)
ggplot(comparison_nn_melt, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Porównanie wyników sieci neuronowych",
       x = "Metryka",
       y = "Wartość") +
  theme_minimal()

#######Klasyfikacja wieloklasowa######
# Wczytanie danych
data_w <- read.csv("breast-cancer-wisconsin.csv", sep=",", header=FALSE)

# Nazwy kolumn
colnames(data_w) <- c("id", "diagnosis", paste0("feature_", 1:(ncol(data_w) - 2)))

# Usunięcie kolumny ID
data_w <- data_w[, -1]

# Konwersja wartości M/B na 2/1 (2 - nowotwór złośliwy, 1 - łagodny)
data_w$diagnosis <- ifelse(data_w$diagnosis == "M", 2, 1)
data_w$diagnosis <- as.factor(data_w$diagnosis)

# Podział na zbiór treningowy i testowy
set.seed(123)
train_idx <- sample(1:nrow(data_w), size = 0.8 * nrow(data_w))
train_data <- data_w[train_idx, -1]  
train_labels <- data_w[train_idx, 1]
test_data <- data_w[-train_idx, -1]
test_labels <- data_w[-train_idx, 1]

# Konwersja do macierzy numerycznej
train_data <- as.matrix(train_data)
test_data <- as.matrix(test_data)

#Cross-Validation 
set.seed(123)
folds_multi <- createFolds(train_labels, k = 5)

# CV dla własnego modelu KNN (wieloklasowa)
custom_knn_multi_cv <- function(k_value) {
  acc_fold <- numeric(length(folds_multi))
  for(i in seq_along(folds_multi)) {
    fold_idx <- folds_multi[[i]]
    x_tr_cv <- train_data[-fold_idx, ]
    y_tr_cv <- train_labels[-fold_idx]
    x_val_cv <- train_data[fold_idx, ]
    y_val_cv <- train_labels[fold_idx]
    
    preds_cv <- knn_wieloklasowa(x_tr_cv, y_tr_cv, x_val_cv, k = k_value)
    acc_fold[i] <- accuracy(preds_cv, as.numeric(as.character(y_val_cv)))
  }
  return(mean(acc_fold))
}

k_values_multi <- seq(1, 20, by = 2)
cv_acc_knn_multi <- sapply(k_values_multi, custom_knn_multi_cv)
best_k_multi <- k_values_multi[which.max(cv_acc_knn_multi)]
cat("CV: Najlepsze k dla własnego KNN (wieloklasowa) =", best_k_multi, "ze średnią accuracy =", max(cv_acc_knn_multi), "\n")

# CV dla własnego drzewa decyzyjnego (wieloklasowa)
custom_tree_multi_cv <- function(depth_value) {
  acc_fold <- numeric(length(folds_multi))
  for(i in seq_along(folds_multi)) {
    fold_idx <- folds_multi[[i]]
    x_tr_cv <- train_data[-fold_idx, ]
    y_tr_cv <- train_labels[-fold_idx]
    x_val_cv <- train_data[fold_idx, ]
    y_val_cv <- train_labels[fold_idx]
    
    tree_mod_cv <- decision_tree_w(x_tr_cv, y_tr_cv, depth = depth_value, criterion = "entropy")
    preds_cv <- predict_tree_w(tree_mod_cv, x_val_cv)
    preds_cv <- factor(preds_cv, levels = levels(train_labels))
    acc_fold[i] <- accuracy(preds_cv, y_val_cv)
  }
  return(mean(acc_fold))
}

depth_values_multi <- 1:10
cv_acc_tree_multi <- sapply(depth_values_multi, custom_tree_multi_cv)
best_depth_multi <- depth_values_multi[which.max(cv_acc_tree_multi)]
cat("CV: Najlepsza głębokość dla własnego drzewa (wieloklasowa) =", best_depth_multi, "ze średnią accuracy =", max(cv_acc_tree_multi), "\n")

# CV dla własnej sieci neuronowej (wieloklasowa)
custom_nn_multi_cv <- function(hidden_neurons, epochs, lr) {
  acc_fold <- numeric(length(folds_multi))
  for(i in seq_along(folds_multi)) {
    fold_idx <- folds_multi[[i]]
    x_tr_cv <- train_data[-fold_idx, ]
    y_tr_cv <- train_labels[-fold_idx]
    x_val_cv <- train_data[fold_idx, ]
    y_val_cv <- train_labels[fold_idx]
    
    nn_mod_cv <- neural_network_train(x_tr_cv, y_tr_cv, hidden_neurons = hidden_neurons, epochs = epochs, learning_rate = lr)
    preds_cv <- predict_nn(nn_mod_cv, x_val_cv)
    acc_fold[i] <- accuracy(preds_cv, y_val_cv)
  }
  return(mean(acc_fold))
}

cv_acc_nn_multi <- custom_nn_multi_cv(hidden_neurons = 5, epochs = 200, lr = 0.01)
cat("CV: Własna sieć NN (wieloklasowa) dla 5 neuronów, 200 epok, lr=0.01 ma średnią accuracy =", cv_acc_nn_multi, "\n")


#Własna implementacja KNN
knn_predict_w <- knn_wieloklasowa(train_data, train_labels, test_data, k = 5)
knn_accuracy <- accuracy(knn_predict_w, as.numeric(as.character(test_labels)))
cat("Dokładność KNN dla k=5 (własne):", knn_accuracy, "\n")
#Analiza wpływU parametru k 
k_values <- seq(1, 20, by = 2)  
knn_accuracies <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  current_k <- k_values[i]
  knn_pred <- knn_wieloklasowa(train_data, train_labels, test_data, k = current_k)
  knn_accuracies[i] <- accuracy(knn_pred, as.numeric(as.character(test_labels)))
  cat("k =", current_k, "dokładność:", knn_accuracies[i], "\n")
}

knn_df <- data.frame(k = k_values, accuracy = knn_accuracies)

ggplot(knn_df, aes(x = k, y = accuracy)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Wpływ parametru k na dokładność KNN", 
       x = "Liczba sąsiadów (k)", y = "Dokładność") +
  theme_minimal()


knn_predict_w <- knn_wieloklasowa(train_data, train_labels, test_data, k = 9)
knn_accuracy <- accuracy(knn_predict_w, as.numeric(as.character(test_labels)))
cat("Dokładność KNN dla k=9 (własne):", knn_accuracy, "\n")

#Wbudowany model KNN
knn_model <- train(train_data, train_labels, method = "knn", tuneGrid = data.frame(k = 9))
knn_model_predictions <- predict(knn_model, test_data)
knn_model_accuracy <- accuracy(knn_model_predictions, test_labels)
cat("Dokładność KNN (wbudowane):", knn_model_accuracy, "\n")

#Własne drzewo decyzyjne
tree_model <- decision_tree_w(train_data, train_labels, depth = 9, criterion = "entropy")
tree_predictions <- predict_tree_w(tree_model, test_data)
tree_predictions <- factor(tree_predictions, levels = levels(test_labels))
tree_accuracy <- accuracy(tree_predictions, test_labels)
cat("Dokładność drzewa decyzyjnego dla k=9 (własne):", tree_accuracy, "\n")

#Analiza dokłądnośći drzewa decyzyjnego w zależności od głębokości 
depth_values <- 1:10  
tree_accuracies <- numeric(length(depth_values))

for (i in seq_along(depth_values)) {
  current_depth <- depth_values[i]
  tree_model_temp <- decision_tree_w(train_data, train_labels, depth = current_depth, criterion = "entropy")
  tree_pred <- predict_tree_w(tree_model_temp, test_data)
  tree_pred <- factor(tree_pred, levels = levels(test_labels))
  tree_accuracies[i] <- accuracy(tree_pred, test_labels)
  cat("Głębokość =", current_depth, "dokładność:", tree_accuracies[i], "\n")
}

tree_df <- data.frame(depth = depth_values, accuracy = tree_accuracies)

ggplot(tree_df, aes(x = depth, y = accuracy)) +
  geom_line(color = "green") +
  geom_point(color = "orange") +
  labs(title = "Wpływ głębokości drzewa na dokładność drzewa decyzyjnego", 
       x = "Głębokość drzewa", y = "Dokładność") +
  theme_minimal()


#Wbudowane drzewo decyzyjne
train_data_dt <- as.data.frame(train_data)
train_data_dt$target <- train_labels
test_data_dt <- as.data.frame(test_data)

tree_model_builtin <- train(target ~ ., data = train_data_dt, method = "rpart", tuneGrid = data.frame(cp = 0.01))
tree_model_predictions <- predict(tree_model_builtin, newdata = test_data_dt)
tree_model_accuracy <- accuracy(tree_model_predictions, test_labels)
cat("Dokładność drzewa decyzyjnego (wbudowane):", tree_model_accuracy, "\n")

#Własne sieci neuronowe
nn_model <- neural_network_train(train_data, train_labels, hidden_neurons = 5, epochs = 200, learning_rate = 0.01)
nn_predictions <- predict_nn(nn_model, test_data)
nn_accuracy <- accuracy(nn_predictions, test_labels)
cat("Dokładność sieci neuronowej dla 5 (własne):", nn_accuracy, "\n")

#Analiza dokłądności sieci w zależności od liczby neuronów
hidden_neurons_values <- seq(5, 10, by = 1)  
nn_accuracies <- numeric(length(hidden_neurons_values))

for (i in seq_along(hidden_neurons_values)) {
  current_hidden <- hidden_neurons_values[i]
  nn_model_temp <- neural_network_train(train_data, train_labels, 
                                        hidden_neurons = current_hidden, 
                                        epochs = 200, 
                                        learning_rate = 0.01)
  nn_pred <- predict_nn(nn_model_temp, test_data)
  nn_accuracies[i] <- accuracy(nn_pred, test_labels)
  cat("Liczba neuronów =", current_hidden, "dokładność:", nn_accuracies[i], "\n")
}

nn_df <- data.frame(hidden_neurons = hidden_neurons_values, accuracy = nn_accuracies)

ggplot(nn_df, aes(x = hidden_neurons, y = accuracy)) +
  geom_line(color = "purple") +
  geom_point(color = "pink") +
  labs(title = "Wpływ liczby neuronów w warstwie ukrytej na dokładność sieci neuronowej", 
       x = "Liczba neuronów", y = "Dokładność") +
  theme_minimal()

nn_model <- neural_network_train(train_data, train_labels, hidden_neurons =C(9), epochs = 200, learning_rate = 0.01)
nn_predictions <- predict_nn(nn_model, test_data)
nn_accuracy <- accuracy(nn_predictions, test_labels)
cat("Dokładność sieci neuronowej dla 9 (własne):", nn_accuracy, "\n")

#Wbudowana sieć neuronowa
train_labels_onehot <- class.ind(as.factor(train_labels))
test_labels_onehot <- class.ind(as.factor(test_labels))

# Trening wbudowanej sieci neuronowej
nn_model_builtin <- nnet(train_data, train_labels_onehot, size = 9, maxit = 200, trace = FALSE)

# Predykcja
nn_model_predictions <- predict(nn_model_builtin, test_data, type = "raw")

# Konwersja predykcji do etykiet klasowych
nn_model_predictions_labels <- max.col(nn_model_predictions)

# Obliczenie dokładności
nn_model_accuracy <- accuracy(nn_model_predictions_labels, as.numeric(test_labels))
cat("Dokładność wbudowanej sieci neuronowej:", nn_model_accuracy, "\n")

#WYKRESY PORÓWNANIA MODELI

#Porównanie KNN_klasyfikacja_wieloklasowa
metrics_knn <- data.frame(
  Model = c("KNN_Własne", "KNN_Wbudowane"),
  Accuracy = c(knn_accuracy, knn_model_accuracy)
)

ggplot(metrics_knn, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Porównanie dokładności: KNN", y = "Dokładność", x = "Model") +
  theme_minimal()

#Porównanie Drzew Decyzyjnych_klasyfikacja_wieloklasowa
metrics_tree <- data.frame(
  Model = c("Drzewo_Własne", "Drzewo_Wbudowane"),
  Accuracy = c(tree_accuracy, tree_model_accuracy)
)

ggplot(metrics_tree, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Porównanie dokładności: Drzewa Decyzyjne", y = "Dokładność", x = "Model") +
  theme_minimal()

#Porównanie Sieci Neuronowyc_klasyfikacja_wieloklasowah
metrics_nn <- data.frame(
  Model = c("NN_Własne", "NN_Wbudowane"),
  Accuracy = c(nn_accuracy, nn_model_accuracy)
)

ggplot(metrics_nn, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Porównanie dokładności: Sieci Neuronowe", y = "Dokładność", x = "Model") +
  theme_minimal()
