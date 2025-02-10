# ===============================
# Ładowanie bibliotek
# ===============================
library(readxl)
library(ggplot2)
library(rpart)
library(caret)
library(nnet)
library(gridExtra)

# ============================================================
# Regresja
# ============================================================

source("funkcje.R")

# Wczytanie danych
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

# Data frame do przechowywania wyników
results_knn_custom <- data.frame(k = numeric(), rmse = numeric(), r2 = numeric())

# Iteracja po różnych wartościach k
for (k_val in k_values) {
  hyperparams_knn_custom <- list(k = k_val, threshold = 0.5)
  
  # Przeprowadzenie 10-krotnej CV dla danego ustawienia hiperparametru
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
  
  # Zapisanie wyników
  results_knn_custom <- rbind(results_knn_custom, 
                              data.frame(k = k_val,
                                         rmse = cv_results["rmse"],
                                         r2 = cv_results["r2"]))
}

# Wyświetlenie wyników tuningu dla własnego modelu k-NN
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

# Wizualizacja wyników – zależność R^2 od liczby sąsiadów k
ggplot(results_tree_custom, aes(x = max_depth, y = r2)) +
  geom_line(color = "blue") +
  geom_point(color = "blue") +
  ggtitle("Własny model k-NN: RMSE w zależności od k") +
  xlab("Liczba najbliższych sąsiadów (k)") +
  ylab(expression(R^2))

# Definicja zakresu wartości hiperparametru max_depth
depth_values <- c(3, 5, 7, 9, 11)

# Data frame do przechowywania wyników
results_tree_custom <- data.frame(max_depth = numeric(), rmse = numeric(), r2 = numeric())

# Iteracja po różnych wartościach max_depth
for (depth_val in depth_values) {
  
  # Ustawienie hiperparametru dla bieżącej iteracji
  hyperparams_tree_custom <- list(max_depth = depth_val)
  
  # Przeprowadzenie 10-krotnej walidacji krzyżowej dla danego max_depth
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
  
  # Zapisanie wyników do data frame
  results_tree_custom <- rbind(results_tree_custom,
                               data.frame(max_depth = depth_val,
                                          rmse = cv_results_tree["rmse"],
                                          r2 = cv_results_tree["r2"]))
}

# Wyświetlenie wyników tuningu dla własnego modelu drzewa decyzyjnego
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

# Data frame do przechowywania wyników tuningu
results_nn_custom <- data.frame(hidden_neurons = numeric(),
                                learning_rate = numeric(),
                                rmse = numeric(),
                                r2 = numeric())

# Iteracja po każdej kombinacji hiperparametrów
for (i in 1:nrow(nn_grid)) {
  h_neurons <- nn_grid$hidden_neurons[i]
  lr <- nn_grid$learning_rate[i]
  
  # Ustawienie hiperparametrów dla bieżącej iteracji
  hyperparams_nn_custom <- list(hidden_neurons = h_neurons,
                                epochs = 2000,
                                learning_rate = lr)
  
  # Przeprowadzenie 10-krotnej walidacji krzyżowej
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
  
  # Zapisanie wyników
  results_nn_custom <- rbind(results_nn_custom,
                             data.frame(hidden_neurons = h_neurons,
                                        learning_rate = lr,
                                        rmse = cv_results_nn["rmse"],
                                        r2 = cv_results_nn["r2"]))
}

# Wyświetlenie wyników tuningu dla własnego modelu sieci neuronowej
print(results_nn_custom)

# Wybór najlepszego modelu na podstawie minimalnego RMSE
best_nn_model <- results_nn_custom[which.min(results_nn_custom$rmse), ]
print(best_nn_model)

cat("Najlepsza konfiguracja sieci neuronowej:\n")
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


# Definicja hiperparametrów jako listy
hyperparams_knn <- list(k = 5)
hyperparams_tree <- list(max_depth = 5)
hyperparams_nn   <- list(hidden_neurons = 50, epochs = 2000, learning_rate = 0.0001)

# Ocena własnych modeli przy użyciu własnej walidacji krzyżowej (10-krotne CV)
# Dla KNN chcemy przekazać hiperparametr do funkcji predykcyjnej, dlatego pass_to_predict = TRUE
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

# Dla drzewa decyzyjnego i sieci neuronowej nie przekazujemy hiperparametrów do funkcji predykcyjnej
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
cat("KNN RMSE:", metrics_knn["rmse"], "R^2:", metrics_knn["r2"], "\n")
cat("Drzewo RMSE:", metrics_tree["rmse"], "R^2:", metrics_tree["r2"], "\n")
cat("Sieć Neuronowa RMSE:", metrics_nn["rmse"], "R^2:", metrics_nn["r2"], "\n")

# Porównanie z modelami z pakietów
ctrl <- trainControl(method = "cv", number = 5)

# Model KNN przy użyciu caret
knn_model_caret <- train(Concrete_Compressive_Strength ~ ., 
                         data = data_reg, 
                         method = "knn", 
                         tuneGrid = data.frame(k = 5),
                         trControl = ctrl)
caret_knn_rmse <- min(knn_model_caret$results$RMSE)
caret_knn_r2   <- min(knn_model_caret$results$Rsquared)

# Model drzewa decyzyjnego przy użyciu rpart
tree_model_caret <- train(Concrete_Compressive_Strength ~ ., 
                          data = data_reg, 
                          method = "rpart", 
                          tuneGrid = data.frame(cp = 0.001),  # możesz dostroić cp
                          trControl = ctrl,
                          control = rpart.control(maxdepth = 5))
caret_tree_rmse <- min(tree_model_caret$results$RMSE)
caret_tree_r2   <- max(tree_model_caret$results$Rsquared)

# Model sieci neuronowej przy użyciu nnet
nn_model_caret <- train(Concrete_Compressive_Strength ~ ., 
                        data = data_reg, 
                        method = "nnet", 
                        tuneGrid = data.frame(size = 50, decay = 0.0001),
                        trControl = ctrl,
                        maxit = 2000,
                        linout = TRUE,
                        trace = TRUE)  # trace=FALSE, żeby nie "zaśmiecać" konsoli
caret_nn_rmse <- min(nn_model_caret$results$RMSE)
caret_nn_r2   <- max(nn_model_caret$results$Rsquared)

cat("\nWyniki modeli pakietowych:\n")
cat("Caret KNN RMSE:", caret_knn_rmse, "R^2:", caret_knn_r2, "\n")
cat("Caret Drzewo RMSE:", caret_tree_rmse, "R^2:", caret_tree_r2, "\n")
cat("Caret Sieć NN RMSE:", caret_nn_rmse, "R^2:", caret_nn_r2, "\n")

# Porównanie wyników CV (własne modele) z modelami pakietowymi
results_comparison_cv <- data.frame(
  Model = rep(c("KNN", "Drzewo Decyzyjne", "Sieć Neuronowa"), 2),
  Podejście = rep(c("Własne", "Pakietowe"), each = 3),
  RMSE = c(metrics_knn["rmse"], metrics_tree["rmse"], metrics_nn["rmse"], 
           caret_knn_rmse, caret_tree_rmse, caret_nn_rmse),
  R2   = c(metrics_knn["r2"], metrics_tree["r2"], metrics_nn["r2"],
           caret_knn_r2, caret_tree_r2, caret_nn_r2)
)

print(results_comparison_cv)

ggplot(results_comparison_cv, aes(x = Model, y = RMSE, fill = Podejście)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Porównanie RMSE (CV 10-krotne): Własne vs Pakietowe") +
  theme_minimal()

ggplot(results_comparison_cv, aes(x = Model, y = R2, fill = Podejście)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Porównanie R^2 (CV 10-krotne): Własne vs Pakietowe") +
  theme_minimal()

# ============================================================
# Problem Klasyfikacji binarnej
# ============================================================

source("funkcje.R")  # Upewnij się, że powyższe funkcje są w pliku funkcje.R

# Wczytanie danych
data <- read.csv("bank.csv", sep = ";")

# Konwersja zmiennej celu na 0/1 (na razie numerycznie)
data$y <- ifelse(data$y == "yes", 1, 0)

# Lista zmiennych kategorycznych
categorical_cols <- c("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome")
for (col in categorical_cols) {
  data[[col]] <- as.factor(data[[col]])
}

# One-hot encoding – tworzymy macierz cech (bez interceptu)
encoded_data <- model.matrix(y ~ . - 1, data = data)
encoded_data <- as.data.frame(encoded_data)
# Dołączamy oryginalną zmienną celu (numeryczną 0/1)
encoded_data$y <- data$y

# Przygotowanie danych dla własnych modeli (numeryczne y)
features_scaled <- as.data.frame(scale(encoded_data[, -ncol(encoded_data)]))
target <- encoded_data$y
data_cls_numeric <- cbind(features_scaled, y = target)
# Sprawdź, że y jest numeryczne
str(data_cls_numeric$y)  # Powinno pokazać 'num [1:...]'

# Przygotowanie danych dla modeli pakietowych (factor y)
# Możesz skopiować dane z wersji numerycznej, a następnie przekonwertować y na factor.
data_cls_factor <- data_cls_numeric
data_cls_factor$y <- as.factor(data_cls_factor$y)
levels(data_cls_factor$y) <- c("X0", "X1")  # Poziomy zaczynają się od litery, co jest wymagane przez caret

names(data_cls_factor) <- make.names(names(data_cls_factor))

cat("Rozkład klas w danych (wszystkie):\n")
print(table(data_cls_factor$y))

# Definicja zakresu wartości hiperparametru k
k_values <- c(3, 5, 7, 9, 11)

# Data frame do przechowywania wyników
results_knn_cls_custom <- data.frame(k = numeric(), accuracy = numeric())

# Iteracja po wartościach k
for (k_val in k_values) {
  hyperparams_knn_cls_custom <- list(k = k_val, threshold = 0.5)
  
  # Przeprowadzenie 10-krotnej walidacji krzyżowej dla danego k
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
  
  # Zapisanie wyniku (Accuracy) dla danego k
  results_knn_cls_custom <- rbind(results_knn_cls_custom,
                                  data.frame(k = k_val,
                                             accuracy = cv_results_knn_cls["accuracy"]))
}

# Wyświetlenie wyników tuningu dla własnego modelu k-NN
print(results_knn_cls_custom)

# Wybór najlepszego modelu – wiersza z maksymalnym accuracy
best_knn_model <- results_knn_cls_custom[which.max(results_knn_cls_custom$accuracy), ]
print(best_knn_model)

cat("Najlepsza konfiguracja dla modelu k-NN (klasyfikacja binarna):\n")
cat("Liczba najbliższych sąsiadów (k):", best_knn_model$k, "\n") # k = 11
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

# Data frame do przechowywania wyników
results_tree_cls_custom <- data.frame(depth = numeric(), accuracy = numeric())

# Iteracja po różnych wartościach depth
for (d in depth_values) {
  hyperparams_tree_cls_custom <- list(depth = d)
  
  # Przeprowadzenie 10-krotnej walidacji krzyżowej dla danego depth
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
  
  # Zapisanie wyniku (Accuracy) dla danego depth
  results_tree_cls_custom <- rbind(results_tree_cls_custom,
                                   data.frame(depth = d,
                                              accuracy = cv_results_tree_cls["accuracy"]))
}

# Wyświetlenie wyników tuningu dla własnego modelu drzewa
print(results_tree_cls_custom)

# Wybór najlepszego modelu – wiersza z maksymalnym accuracy
best_tree_model <- results_tree_cls_custom[which.max(results_tree_cls_custom$accuracy), ]
print(best_tree_model)

cat("Najlepsza konfiguracja drzewa decyzyjnego (klasyfikacja binarna):\n")
cat("Maksymalna głębokość:", best_tree_model$depth, "\n") # depth = 3
cat("Osiągnięty Accuracy:", best_tree_model$accuracy, "\n")


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

# Data frame do przechowywania wyników
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
  
  # Przeprowadzenie 10-krotnej walidacji krzyżowej dla danej kombinacji hiperparametrów
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
  
  # Zapisanie wyniku (Accuracy) dla danej kombinacji
  results_nn_cls_custom <- rbind(results_nn_cls_custom,
                                 data.frame(hidden_neurons = h_neurons,
                                            learning_rate = lr,
                                            accuracy = cv_results_nn_cls["accuracy"]))
}

# Wyświetlenie wyników tuningu dla własnego modelu sieci neuronowej
print(results_nn_cls_custom)

# Wybór najlepszego modelu – wiersza z maksymalnym accuracy
best_model <- results_nn_cls_custom[which.max(results_nn_cls_custom$accuracy), ]
print(best_model)

# Możesz też wyświetlić komunikat z najlepszą konfiguracją:
cat("Najlepsza konfiguracja sieci neuronowej:\n")
cat("Liczba neuronów:", best_model$hidden_neurons, "\n") # 70
cat("Learning rate:", best_model$learning_rate, "\n") # 0.0005
cat("Osiągnięty Accuracy:", best_model$accuracy, "\n")

# Wizualizacja wyników – Accuracy w zależności od liczby neuronów i współczynnika uczenia
ggplot(results_nn_cls_custom, aes(x = factor(hidden_neurons), y = accuracy, fill = factor(learning_rate))) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Własny model sieci neuronowej (klasyfikacja binarna): Accuracy") +
  xlab("Liczba neuronów w warstwie ukrytej") +
  ylab("Accuracy") +
  labs(fill = "Learning Rate")

# Definicja hiperparametrów dla modeli klasyfikacyjnych
hyperparams_knn_cls <- list(k = 10, threshold = 0.5)
hyperparams_tree_cls <- list(depth = 5)
hyperparams_nn_cls   <- list(hidden_neurons = 70, epochs = 2000, learning_rate = 0.0005)

set.seed(123)

# Użycie funkcji custom_cv_multi dla własnych modeli klasyfikacyjnych
metrics_knn_cls <- custom_cv(
  data = data_cls_numeric,
  k_folds = 5,
  train_func = train_knn_classification,
  predict_func = predict_knn_classification,
  performance_funcs = list(accuracy = accuracy),
  hyperparams = hyperparams_knn_cls,
  target_col = "y",
  pass_to_predict = TRUE
)

metrics_tree_cls <- custom_cv(
  data = data_cls_numeric,
  k_folds = 5,
  train_func = train_tree_classification,
  predict_func = predict_tree_classification,
  performance_funcs = list(accuracy = accuracy),
  hyperparams = hyperparams_tree_cls,
  target_col = "y",
  pass_to_predict = FALSE
)

metrics_nn_cls <- custom_cv(
  data = data_cls_numeric,
  k_folds = 5,
  train_func = train_nn_classification,
  predict_func = predict_nn_classification,
  performance_funcs = list(accuracy = accuracy),
  hyperparams = hyperparams_nn_cls,
  target_col = "y",
  pass_to_predict = FALSE
)

cat("Wyniki CV dla własnych modeli (klasyfikacja):\n")
cat("k-NN Accuracy:", metrics_knn_cls["accuracy"], "\n")
cat("Drzewo Accuracy:", metrics_tree_cls["accuracy"], "\n")
cat("Sieć Neuronowa Accuracy:", metrics_nn_cls["accuracy"], "\n")

# Przygotowanie zbiorów danych dla modeli pakietowych
set.seed(123)
train_indices <- createDataPartition(data_cls_factor$y, p = 0.8, list = FALSE)
train_data_cls <- data_cls_factor[train_indices, ]
test_data_cls  <- data_cls_factor[-train_indices, ]
cat("Rozkład klas w zbiorze treningowym:\n")
print(table(train_data_cls$y))
cat("Rozkład klas w zbiorze testowym:\n")
print(table(test_data_cls$y))


# 6. Ustawienia kontrolne dla modeli pakietowych (caret)
# Użyjemy stratygowanych foldów – funkcja createFolds gwarantuje, że w każdym foldzie będą obie klasy.
folds <- createFolds(train_data_cls$y, k = 5)
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, index = folds)
lapply(folds, function(idx) table(train_data_cls$y[idx]))


# Model k-NN przy użyciu caret
knn_model_pkg <- train(y ~ ., 
                       data = train_data_cls, 
                       method = "knn", 
                       tuneGrid = data.frame(k = 10),
                       trControl = ctrl)

knn_predictions_pkg <- predict(knn_model_pkg, newdata = test_data_cls)
knn_acc_pkg <- mean(knn_predictions_pkg == test_data_cls$y)

# --- Model drzewa decyzyjnego (rpart) ---
# Używamy metody "rpart". Aby ustawić maksymalną głębokość drzewa, przekazujemy
# kontrolę za pomocą rpart.control(maxdepth = 5).

# --- Model drzewa decyzyjnego przy użyciu caret ---
# Problem przy rpart często wynika z bardzo niskiej wartości cp lub braku próbki jednej klasy w foldach.
# Dlatego użyjemy metody "rpart" z ustalonym parametrem cp (tutaj 0.001) oraz dodatkowych ustawień:
tree_model_pkg <- train(
  y ~ ., 
  data = train_data_cls, 
  method = "rpart", 
  tuneGrid = data.frame(cp = 0.001),  
  trControl = ctrl,
  control = rpart.control(maxdepth = 5),
  metric = "Accuracy"
)
tree_predictions_pkg <- predict(tree_model_pkg, newdata = test_data_cls)
tree_acc_pkg <- mean(tree_predictions_pkg == test_data_cls$y)

# --- Model sieci neuronowej (nnet) ---
nn_model_pkg <- train(y ~ ., 
                      data = train_data_cls, 
                      method = "nnet", 
                      tuneGrid = data.frame(size = 50, decay = 0.0005),
                      trControl = ctrl,
                      maxit = 2000,
                      linout = FALSE,
                      trace = TRUE,
                      MaxNWts = 10000)
nn_predictions_pkg <- predict(nn_model_pkg, newdata = test_data_cls)
nn_acc_pkg <- mean(nn_predictions_pkg == test_data_cls$y)

cat("\nWyniki modeli pakietowych (klasyfikacja):\n")
cat("Caret k-NN Accuracy:", knn_acc_pkg, "\n")
cat("Caret Drzewo Accuracy:", tree_acc_pkg, "\n")
cat("Caret Sieć NN Accuracy:", nn_acc_pkg, "\n")

results_comparison_cls <- data.frame(
  Model = rep(c("k-NN", "Drzewo Decyzyjne", "Sieć Neuronowa"), 2),
  Podejście = rep(c("Własne", "Pakietowe"), each = 3),
  Accuracy = c(metrics_knn_cls["accuracy"], metrics_tree_cls["accuracy"], metrics_nn_cls["accuracy"],
               knn_acc_pkg, tree_acc_pkg, nn_acc_pkg)
)

print(results_comparison_cls)

ggplot(results_comparison_cls, aes(x = Model, y = Accuracy, fill = Podejście)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Porównanie dokładności modeli: Własne vs. Pakietowe (klasyfikacja)") +
  theme_minimal()


# ============================================================
# Problem Klasyfikacji Wieloklasowej
# ============================================================

source("funkcje.R")

data <- read.csv("winequality-red.csv", sep = ";")

unique(data$quality)

# Histogram dla zmiennej targetowanej (quality)
ggplot(data, aes(x = quality)) +
  geom_bar(fill = "steelblue", color = "black") +
  ggtitle("Rozkład zmiennej targetowanej (quality)") +
  xlab("Jakość wina") +
  ylab("Liczba próbek") +
  theme_minimal()

# Histogramy  dla cech  numerycznych
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

# Boxploty dla każdej cechy w zależności od jakości wina
feature_boxplots <- lapply(feature_names, function(feature) {
  ggplot(data, aes(x = factor(quality), y = .data[[feature]])) +
    geom_boxplot(fill = "orange", color = "black", alpha = 0.7) +
    ggtitle(paste("Boxplot cechy:", feature, "względem jakości wina")) +
    xlab("Jakość wina") +
    ylab(feature) +
    theme_minimal()
})

# Wyświetlenie kilku boxplotów
do.call(grid.arrange, c(feature_boxplots[1:4], ncol = 2))


# Przygotowanie danych
target <- data$quality
features <- data[, -ncol(data)]

# Standaryzacja cech
features_scaled <- as.data.frame(scale(features))

data_multi <- cbind(features_scaled, quality = target)

# Analiza hiperparametrów dla modeli wieloklasowych

# Założenie: Dane zostały przygotowane w obiekcie data_multi,
# gdzie:
# - cechy zostały znormalizowane (features_scaled)
# - kolumna target to "quality" (liczba jakości wina)
# - zmienna quality w data_multi pozostaje numeryczna – przy wywołaniu custom_cv
#   nasze funkcje (train_knn_multiclass, train_tree_multiclass, train_nn_multiclass)
#   operują na danych numerycznych, a ocena Accuracy odbywa się na podstawie oryginalnych wartości.
#
# Upewnij się, że w pliku funkcje.R masz zdefiniowane:
#   - custom_cv
#   - train_knn_multiclass, predict_knn_multiclass
#   - train_tree_multiclass, predict_tree_multiclass
#   - train_nn_multiclass, predict_nn_multiclass
#   - accuracy (funkcja oceniająca Accuracy)
#
# W poniższych przykładach przyjmujemy 5-krotną walidację krzyżową.

### 1. Tuning dla własnego modelu k-NN (klasyfikacja wieloklasowa)
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
                                               accuracy = cv_results_knn_multi["accuracy"]))
}

print(results_knn_multi_custom)

# Wybór najlepszego modelu k-NN
best_knn_multi <- results_knn_multi_custom[which.max(results_knn_multi_custom$accuracy), ]
cat("Najlepsza konfiguracja k-NN:\n")
cat("Liczba najbliższych sąsiadów (k):", best_knn_multi$k, "\n") # k = 11
cat("Osiągnięta Accuracy:", best_knn_multi$accuracy, "\n\n")


# Wizualizacja wyników dla k-NN
ggplot(results_knn_multi_custom, aes(x = k, y = accuracy)) +
  geom_line() +
  geom_point() +
  ggtitle("Własny model k-NN (klasyfikacja wieloklasowa): Accuracy vs. k") +
  xlab("Liczba najbliższych sąsiadów (k)") +
  ylab("Accuracy")

### 2. Tuning dla własnego modelu drzewa decyzyjnego (klasyfikacja wieloklasowa)
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
                                                accuracy = cv_results_tree_multi["accuracy"]))
}

print(results_tree_multi_custom)

# Wybór najlepszego modelu drzewa
best_tree_multi <- results_tree_multi_custom[which.max(results_tree_multi_custom$accuracy), ]
cat("Najlepsza konfiguracja drzewa decyzyjnego:\n")
cat("Głębokość drzewa (depth):", best_tree_multi$depth, "\n") # depth = 5
cat("Osiągnięta Accuracy:", best_tree_multi$accuracy, "\n\n")

# Wizualizacja wyników dla drzewa
ggplot(results_tree_multi_custom, aes(x = depth, y = accuracy)) +
  geom_line(color = "blue") +
  geom_point(color = "blue") +
  ggtitle("Własny model drzewa decyzyjnego (klasyfikacja wieloklasowa): Accuracy vs. Depth") +
  xlab("Głębokość drzewa (depth)") +
  ylab("Accuracy")

### 3. Tuning dla własnego modelu sieci neuronowej (klasyfikacja wieloklasowa)
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
                                              accuracy = cv_results_nn_multi["accuracy"]))
}

print(results_nn_multi_custom)

# Wybór najlepszego modelu sieci neuronowej
best_nn_multi <- results_nn_multi_custom[which.max(results_nn_multi_custom$accuracy), ]
cat("Najlepsza konfiguracja sieci neuronowej:\n")
cat("Liczba neuronów:", best_nn_multi$hidden_neurons, "\n") # 100 neuronów
cat("Learning rate:", best_nn_multi$learning_rate, "\n") # 0.001 learning rate
cat("Osiągnięta Accuracy:", best_nn_multi$accuracy, "\n")

# Wizualizacja wyników dla sieci neuronowej
ggplot(results_nn_multi_custom, aes(x = factor(hidden_neurons), y = accuracy, fill = factor(learning_rate))) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Własny model sieci neuronowej (klasyfikacja wieloklasowa): Accuracy") +
  xlab("Liczba neuronów w warstwie ukrytej") +
  ylab("Accuracy") +
  labs(fill = "Learning Rate")


# Parametry modeli
k <- 11  # l. sąsiadów dla KNN
max_depth <- 5  # max głębokość drzewa
hidden_neurons <- 100  # l. neuronów ukrytych
epochs <- 2000  # l. epok dla sieci neuronowej
learning_rate <- 0.001 # wsp. uczenia

set.seed(123)

# Modele własne
metrics_knn_multi <- custom_cv(
  data = data_multi,
  k_folds = 5,
  train_func = train_knn_multiclass,
  predict_func = predict_knn_multiclass,
  performance_funcs = list(accuracy = accuracy),
  hyperparams = list(k = k),   # przekazujemy parametr k
  target_col = "quality",
  pass_to_predict = TRUE       # jeśli funkcja predykcyjna również potrzebuje parametru k
)

metrics_tree_multi <- custom_cv(
  data = data_multi,
  k_folds = 5,
  train_func = train_tree_multiclass,
  predict_func = predict_tree_multiclass,
  performance_funcs = list(accuracy = accuracy),
  hyperparams = list(depth = max_depth),  # Używamy "depth", bo tak definiuje funkcja
  target_col = "quality",
  pass_to_predict = FALSE
)

metrics_nn_multi <- custom_cv(
  data = data_multi,
  k_folds = 5,
  train_func = train_nn_multiclass,
  predict_func = predict_nn_multiclass,
  performance_funcs = list(accuracy = accuracy),
  hyperparams = list(hidden_neurons = hidden_neurons, epochs = epochs, learning_rate = learning_rate),
  target_col = "quality",
  pass_to_predict = FALSE
)

cat("Wyniki CV dla własnych modeli (klasyfikacja wieloklasowa):\n")
cat("KNN Accuracy:", metrics_knn_multi["accuracy"], "\n")
cat("Drzewo Accuracy:", metrics_tree_multi["accuracy"], "\n")
cat("Sieć Neuronowa Accuracy:", metrics_nn_multi["accuracy"], "\n")

# Modele pakietowe (wieloklasowe)

# Przygotowanie danych – dla modeli pakietowych target (quality) traktujemy jako factor:
train_indices <- createDataPartition(data_multi$quality, p = 0.8, list = FALSE)
train_data_pkg_multi <- cbind(features_scaled[train_indices, ], quality = as.factor(target[train_indices]))
test_data_pkg_multi  <- cbind(features_scaled[-train_indices, ], quality = as.factor(target[-train_indices]))

# Upewnij się, że poziomy targetu są poprawne (np. zaczynają się od litery)
levels(train_data_pkg_multi$quality) <- make.names(levels(train_data_pkg_multi$quality))
levels(test_data_pkg_multi$quality)  <- make.names(levels(test_data_pkg_multi$quality))

# Ustawienia kontrolne dla caret – 10-krotna CV
ctrl <- trainControl(method = "cv", number = 5)

# Model KNN przy użyciu caret
knn_model_pkg_multi <- train(quality ~ ., 
                             data = train_data_pkg_multi, 
                             method = "knn", 
                             tuneGrid = data.frame(k = k),
                             trControl = ctrl)
knn_predictions_pkg_multi <- predict(knn_model_pkg_multi, newdata = test_data_pkg_multi)
knn_acc_pkg_multi <- mean(knn_predictions_pkg_multi == test_data_pkg_multi$quality)

# Model drzewa decyzyjnego przy użyciu rpart
tree_model_pkg_multi_cv <- train(quality ~ ., 
                                 data = train_data_pkg_multi, 
                                 method = "rpart", 
                                 tuneGrid = data.frame(cp = 0.01),  # lub inna wartość, którą chcesz dostroić
                                 trControl = ctrl,
                                 control = rpart.control(maxdepth = max_depth))
tree_predictions_pkg_multi <- predict(tree_model_pkg_multi, newdata = test_data_pkg_multi, type = "class")
tree_acc_pkg_multi <- mean(tree_predictions_pkg_multi == test_data_pkg_multi$quality)

# Model sieci neuronowej przy użyciu nnet
nn_model_pkg_multi_cv <- train(quality ~ ., 
                               data = train_data_pkg_multi, 
                               method = "nnet", 
                               tuneGrid = data.frame(size = hidden_neurons, decay = learning_rate),
                               trControl = ctrl,
                               maxit = epochs,
                               linout = FALSE,
                               trace = TRUE,
                               MaxNWts = 20000)
nn_predictions_pkg_multi_cv <- predict(nn_model_pkg_multi_cv, newdata = test_data_pkg_multi)
nn_acc_pkg_multi_cv <- mean(nn_predictions_pkg_multi_cv == test_data_pkg_multi$quality)


cat("\nWyniki modeli pakietowych (wieloklasowych):\n")
cat("Caret KNN Accuracy:", knn_acc_pkg_multi, "\n")
cat("Caret Drzewo Accuracy:", tree_acc_pkg_multi, "\n")
cat("Caret Sieć NN Accuracy:", nn_acc_pkg_multi, "\n")

# Porównanie modeli: Własne vs Pakietowe

results_multiclass <- data.frame(
  Model = rep(c("KNN", "Drzewo Decyzyjne", "Sieć Neuronowa"), 2),
  Podejście = rep(c("Własne", "Pakietowe"), each = 3),
  Accuracy = c(metrics_knn_multi["accuracy"], metrics_tree_multi["accuracy"], metrics_nn_multi["accuracy"],
               knn_acc_pkg_multi, tree_acc_pkg_multi, nn_acc_pkg_multi)
)

print(results_multiclass)

ggplot(results_multiclass, aes(x = Model, y = Accuracy, fill = Podejście)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Porównanie dokładności modeli wieloklasowych: Własne vs Pakietowe") +
  theme_minimal()
