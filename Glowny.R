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

# Definicja hiperparametrów jako listy
hyperparams_knn <- list(k = 10)
hyperparams_tree <- list(max_depth = 5)
hyperparams_nn   <- list(hidden_neurons = 50, epochs = 2000, learning_rate = 0.0001)

set.seed(123)

# Ocena własnych modeli przy użyciu własnej walidacji krzyżowej (10-krotne CV)
# Dla KNN chcemy przekazać hiperparametr do funkcji predykcyjnej, dlatego pass_to_predict = TRUE
metrics_knn <- custom_cv(
  data = data_reg,
  k_folds = 10,
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
  k_folds = 10,
  train_func = train_tree,
  predict_func = predict_tree,
  performance_funcs = list(rmse = rmse, r2 = r2_score),
  hyperparams = hyperparams_tree,
  target_col = "Concrete_Compressive_Strength",
  pass_to_predict = FALSE
)

metrics_nn <- custom_cv(
  data = data_reg,
  k_folds = 10,
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
ctrl <- trainControl(method = "cv", number = 10)

# Model KNN przy użyciu caret
knn_model_caret <- train(Concrete_Compressive_Strength ~ ., 
                         data = data_reg, 
                         method = "knn", 
                         tuneGrid = data.frame(k = 10),
                         trControl = ctrl)
caret_knn_rmse <- min(knn_model_caret$results$RMSE)
caret_knn_r2   <- min(knn_model_caret$results$Rsquared)

# Model drzewa decyzyjnego przy użyciu rpart
tree_model_caret <- rpart(Concrete_Compressive_Strength ~ ., 
                          data = data_reg, 
                          control = rpart.control(maxdepth = 5))
caret_tree_predictions <- predict(tree_model_caret, newdata = data_reg)
caret_tree_rmse <- rmse(data_reg$Concrete_Compressive_Strength, caret_tree_predictions)
caret_tree_r2   <- r2_score(data_reg$Concrete_Compressive_Strength, caret_tree_predictions)

# Model sieci neuronowej przy użyciu nnet
nn_model_caret <- nnet(Concrete_Compressive_Strength ~ ., 
                       data = data_reg, 
                       size = 50, maxit = 2000, decay = 0.0001, linout = TRUE, trace = TRUE)
caret_nn_predictions <- predict(nn_model_caret, newdata = data_reg)
caret_nn_rmse <- rmse(data_reg$Concrete_Compressive_Strength, caret_nn_predictions)
caret_nn_r2   <- r2_score(data_reg$Concrete_Compressive_Strength, caret_nn_predictions)

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

# -------------------------------
# Przygotowanie danych dla własnych modeli (numeryczne y)
# -------------------------------
features_scaled <- as.data.frame(scale(encoded_data[, -ncol(encoded_data)]))
target <- encoded_data$y
data_cls_numeric <- cbind(features_scaled, y = target)
# Sprawdź, że y jest numeryczne
str(data_cls_numeric$y)  # Powinno pokazać 'num [1:...]'

# -------------------------------
# Przygotowanie danych dla modeli pakietowych (factor y)
# -------------------------------
# Możesz skopiować dane z wersji numerycznej, a następnie przekonwertować y na factor.
data_cls_factor <- data_cls_numeric
data_cls_factor$y <- as.factor(data_cls_factor$y)
levels(data_cls_factor$y) <- c("X0", "X1")  # Poziomy zaczynają się od litery, co jest wymagane przez caret

names(data_cls_factor) <- make.names(names(data_cls_factor))

cat("Rozkład klas w danych (wszystkie):\n")
print(table(data_cls_factor$y))

# Definicja hiperparametrów dla modeli klasyfikacyjnych
hyperparams_knn_cls <- list(k = 10, threshold = 0.5)
hyperparams_tree_cls <- list(depth = 5)
hyperparams_nn_cls   <- list(hidden_neurons = 50, epochs = 2000, learning_rate = 0.0005)

set.seed(123)

# Użycie funkcji custom_cv_multi dla własnych modeli klasyfikacyjnych
metrics_knn_cls <- custom_cv(
  data = data_cls_numeric,
  k_folds = 10,
  train_func = train_knn_classification,
  predict_func = predict_knn_classification,
  performance_funcs = list(accuracy = accuracy),
  hyperparams = hyperparams_knn_cls,
  target_col = "y",
  pass_to_predict = TRUE
)

metrics_tree_cls <- custom_cv(
  data = data_cls_numeric,
  k_folds = 10,
  train_func = train_tree_classification,
  predict_func = predict_tree_classification,
  performance_funcs = list(accuracy = accuracy),
  hyperparams = hyperparams_tree_cls,
  target_col = "y",
  pass_to_predict = FALSE
)

metrics_nn_cls <- custom_cv(
  data = data_cls_numeric,
  k_folds = 10,
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
folds <- createFolds(train_data_cls$y, k = 10)
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, index = folds)
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
# Wczytanie bibliotek


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

# Podział na zbiór treningowy i testowy (80/20)
set.seed(123)
train_indices <- sample(1:nrow(features), 0.8 * nrow(features))
train_X <- features_scaled[train_indices, ]
train_y <- target[train_indices]
test_X <- features_scaled[-train_indices, ]
test_y <- target[-train_indices]

# Parametry modeli
k <- 5  # l. sąsiadów dla KNN
max_depth <- 10  # max głębokość drzewa
hidden_neurons <- 100  # l. neuronów ukrytych
epochs <- 5000  # l. epok dla sieci neuronowej
learning_rate <- 0.0001 # wsp. uczenia

# Trening modeli klasyfikacyjnych
knn_model <- train_knn_multiclass(train_X, train_y)
tree_model <- train_tree_multiclass(train_X, train_y, max_depth)
nn_model <- train_nn_multiclass(train_X, train_y, hidden_neurons, epochs, learning_rate)

# Predykcja modeli
knn_predictions <- predict_knn_multiclass(knn_model, test_X, k)
tree_predictions <- predict_tree_multiclass(tree_model, test_X)
nn_predictions <- predict_nn_multiclass(nn_model, test_X)


# Funkcja do obliczania dokładności
accuracy <- function(actual, predicted) {
  sum(actual == predicted) / length(actual)
}

# Dokładność dla każdego modelu
knn_acc <- accuracy(test_y, knn_predictions)
tree_acc <- accuracy(test_y, tree_predictions)
nn_acc <- accuracy(test_y, nn_predictions)

cat("KNN Accuracy:", knn_acc, "\n")
cat("Drzewo Decyzyjne Accuracy:", tree_acc, "\n")
cat("Sieć Neuronowa Accuracy:", nn_acc, "\n")

# Macierz pomyłek
confusion_matrix <- function(actual, predicted) {
  table(Predicted = predicted, Actual = actual)
}

cat("KNN Confusion Matrix:\n")
print(confusion_matrix(test_y, knn_predictions))

cat("Drzewo Decyzyjne Confusion Matrix:\n")
print(confusion_matrix(test_y, tree_predictions))

cat("Sieć Neuronowa Confusion Matrix:\n")
print(confusion_matrix(test_y, nn_predictions))

# Wizualizacja
results <- data.frame(
  Model = c("KNN", "Drzewo Decyzyjne", "Sieć Neuronowa"),
  Accuracy = c(knn_acc, tree_acc, nn_acc)
)

ggplot(results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", color = "black") +
  ggtitle("Porównanie dokładności modeli klasyfikacyjnych") +
  theme_minimal()

# Wizualizacja macierzy pomyłek
plot_confusion_matrix <- function(actual, predicted, model_name) {
  cm <- as.data.frame(table(Predicted = predicted, Actual = actual))
  
  ggplot(cm, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "black") +
    scale_fill_gradient(low = "white", high = "blue") +
    geom_text(aes(label = Freq), vjust = 1.5, color = "black", size = 5) +
    ggtitle(paste("Macierz Pomyłek -", model_name)) +
    theme_minimal()
}

plot_confusion_matrix(test_y, knn_predictions, "KNN")
plot_confusion_matrix(test_y, tree_predictions, "Drzewo Decyzyjne")
plot_confusion_matrix(test_y, nn_predictions, "Sieć Neuronowa")

# Modele pakietowe (wieloklasowe)

train_data_pkg_multi <- cbind(train_X, quality = as.factor(train_y))
test_data_pkg_multi  <- cbind(test_X,  quality = as.factor(test_y))

levels(train_data_pkg_multi$quality) <- make.names(levels(train_data_pkg_multi$quality))
levels(test_data_pkg_multi$quality)  <- make.names(levels(test_data_pkg_multi$quality))

# Model KNN przy użyciu caret
set.seed(123)
ctrl_multi <- trainControl(method = "cv", number = 10)
knn_model_pkg_multi <- train(quality ~ ., 
                             data = train_data_pkg_multi, 
                             method = "knn", 
                             tuneGrid = data.frame(k = k),
                             trControl = ctrl_multi)
knn_predictions_pkg_multi <- predict(knn_model_pkg_multi, newdata = test_data_pkg_multi)
knn_acc_pkg_multi <- mean(knn_predictions_pkg_multi == test_data_pkg_multi$quality)

# Drzewo decyzyjne przy użyciu rpart
tree_model_pkg_multi <- rpart::rpart(quality ~ ., 
                                     data = train_data_pkg_multi, 
                                     method = "class", 
                                     control = rpart.control(maxdepth = max_depth))
tree_predictions_pkg_multi <- predict(tree_model_pkg_multi, newdata = test_data_pkg_multi, type = "class")
tree_acc_pkg_multi <- mean(tree_predictions_pkg_multi == test_data_pkg_multi$quality)

# Sieć neuronowa przy użyciu nnet
x_train_multi <- as.matrix(train_data_pkg_multi[, -ncol(train_data_pkg_multi)])
y_train_multi <- class.ind(train_data_pkg_multi$quality)
nn_model_pkg_multi <- nnet::nnet(x_train_multi, y_train_multi,
                                 size = hidden_neurons, 
                                 maxit = epochs, 
                                 decay = learning_rate, 
                                 trace = FALSE,
                                 MaxNWts = 20000)
x_test_multi <- as.matrix(test_data_pkg_multi[, -ncol(test_data_pkg_multi)])
nn_probs_pkg_multi <- predict(nn_model_pkg_multi, newdata = x_test_multi, type = "raw")
# Dla każdej obserwacji wybieramy indeks maksymalnego prawdopodobieństwa
nn_pred_indices <- apply(nn_probs_pkg_multi, 1, which.max)
# Mapujemy indeksy na etykiety (poziomy faktora)
nn_predictions_pkg_multi <- factor(nn_pred_indices, labels = levels(test_data_pkg_multi$quality))
nn_acc_pkg_multi <- mean(nn_predictions_pkg_multi == test_data_pkg_multi$quality)

cat("Modele pakietowe (wieloklasowe):\n")
cat("KNN Accuracy:", knn_acc_pkg_multi, "\n")
cat("Drzewo Decyzyjne Accuracy:", tree_acc_pkg_multi, "\n")
cat("Sieć Neuronowa Accuracy:", nn_acc_pkg_multi, "\n")

# Porównanie modeli: Własne vs Pakietowe

results_multiclass <- data.frame(
  Model = rep(c("KNN", "Drzewo Decyzyjne", "Sieć Neuronowa"), 2),
  Podejście = rep(c("Własne", "Pakietowe"), each = 3),
  Accuracy = c(knn_acc, tree_acc, nn_acc, knn_acc_pkg_multi, tree_acc_pkg_multi, nn_acc_pkg_multi)
)

print(results_multiclass)

# Wykres porównawczy dokładności
ggplot(results_multiclass, aes(x = Model, y = Accuracy, fill = Podejście)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Porównanie dokładności modeli wieloklasowych: Własne vs Pakietowe") +
  theme_minimal()

# Opcjonalnie: wyświetlenie macierzy pomyłek dla modeli pakietowych
cat("KNN Confusion Matrix (pakietowe):\n")
print(table(Predicted = knn_predictions_pkg_multi, Actual = test_data_pkg_multi$quality))
cat("Drzewo Decyzyjne Confusion Matrix (pakietowe):\n")
print(table(Predicted = tree_predictions_pkg_multi, Actual = test_data_pkg_multi$quality))
cat("Sieć Neuronowa Confusion Matrix (pakietowe):\n")
print(table(Predicted = nn_predictions_pkg_multi, Actual = test_data_pkg_multi$quality))

