# Wczytanie bibliotek 
library(readxl)
library(ggplot2)
library(rpart)
library(caret)
library(nnet)
library(gridExtra)
# ============================================================
# Problem Regresji
# ===========================================9=================
source("funkcje.R")

# Wczytanie danych
data <- read_excel("Concrete_Data.xls")

# Uproszczenie nazw kolumn
colnames(data) <- c(
  "Cement",
  "Blast_Furnace_Slag",
  "Fly_Ash",
  "Water",
  "Superplasticizer",
  "Coarse_Aggregate",
  "Fine_Aggregate",
  "Age",
  "Concrete_Compressive_Strength"
)

print(colnames(data))

# Zmienna celu i cechy
target <- data$Concrete_Compressive_Strength
features <- data[, -ncol(data)]  
# Standaryzacja cech (zamiast normalizacji)
features_scaled <- as.data.frame(scale(features))

# Podział na zbiór treningowy i testowy (80/20)
set.seed(123)
train_indices <- sample(1:nrow(features), 0.8 * nrow(features))
train_X <- features_scaled[train_indices, ]
train_y <- target[train_indices]
test_X <- features_scaled[-train_indices, ]
test_y <- target[-train_indices]

# Trening modeli

# KNN
knn_model <- train_knn(train_X, train_y)

# Drzewo decyzyjne
tree_model <- train_tree(train_X, train_y, 5)

# Sieć neuronowa 
nn_model <- train_nn(train_X, train_y, 50, 2000, 0.0005)

# Predykcja modeli

knn_predictions <- predict_knn(knn_model, test_X, 10)
tree_predictions <- predict_tree(tree_model, test_X)

# Sieć neuronowa – predykcja ze standaryzacją 
nn_predictions <- predict_nn(nn_model, test_X)

# Ocena modeli

knn_rmse <- rmse(test_y, knn_predictions)
tree_rmse <- rmse(test_y, tree_predictions)
nn_rmse <- rmse(test_y, nn_predictions)

knn_r2 <- r2_score(test_y, knn_predictions)
tree_r2 <- r2_score(test_y, tree_predictions)
nn_r2 <- r2_score(test_y, nn_predictions)

cat("KNN RMSE:", knn_rmse, "R^2:", knn_r2, "\n")
cat("Drzewo Decyzyjne RMSE:", tree_rmse, "R^2:", tree_r2, "\n")
cat("Sieć Neuronowa RMSE:", nn_rmse, "R^2:", nn_r2, "\n")

# Wizualizacja wyników
results <- data.frame(
  Model = c("KNN", "Drzewo Decyzyjne", "Sieć Neuronowa"),
  RMSE = c(knn_rmse, tree_rmse, nn_rmse),
  R2 = c(knn_r2, tree_r2, nn_r2)
)

# Wykres RMSE
ggplot(results, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", color = "black") +
  ggtitle("Porównanie RMSE modeli") +
  theme_minimal()

# Wykres R^2
ggplot(results, aes(x = Model, y = R2, fill = Model)) +
  geom_bar(stat = "identity", color = "black") +
  ggtitle("Porównanie R^2 modeli") +
  theme_minimal()

# Łączymy standaryzowane cechy z zmienną celu
data_reg <- cbind(features_scaled, Concrete_Compressive_Strength = target)

# Podział na zbiór treningowy i testowy (analogicznie jak wcześniej)
set.seed(123)
train_indices <- sample(1:nrow(data_reg), 0.8 * nrow(data_reg))
train_data <- data_reg[train_indices, ]
test_data  <- data_reg[-train_indices, ]



# Używamy walidacji krzyżowej (10-krotna CV)
ctrl <- trainControl(method = "cv", number = 10)

# Trening modelu KNN (regresja) – przyjmujemy k = 10
knn_model_caret <- train(Concrete_Compressive_Strength ~ ., 
                         data = train_data, 
                         method = "knn", 
                         tuneGrid = data.frame(k = 10),
                         trControl = ctrl)

# Predykcja oraz ocena
knn_predictions_caret <- predict(knn_model_caret, newdata = test_data)
caret_knn_rmse <- RMSE(knn_predictions_caret, test_data$Concrete_Compressive_Strength)
caret_knn_r2   <- R2(knn_predictions_caret, test_data$Concrete_Compressive_Strength)

# Trening drzewa decyzyjnego z maksymalną głębokością = 5
tree_model_caret <- rpart::rpart(Concrete_Compressive_Strength ~ ., 
                                 data = train_data, 
                                 control = rpart.control(maxdepth = 5))

tree_predictions_caret <- predict(tree_model_caret, newdata = test_data)
caret_tree_rmse <- RMSE(tree_predictions_caret, test_data$Concrete_Compressive_Strength)
caret_tree_r2   <- R2(tree_predictions_caret, test_data$Concrete_Compressive_Strength)

nn_model_caret <- nnet(train_data[, -ncol(train_data)], train_data$Concrete_Compressive_Strength, 
                       size = 50, maxit = 2000, decay = 0.0005, linout = TRUE, trace = FALSE)

# Predykcja oraz ocena
nn_predictions_caret <- predict(nn_model_caret, test_data[, -ncol(test_data)])
caret_nn_rmse <- RMSE(nn_predictions_caret, test_data$Concrete_Compressive_Strength)
caret_nn_r2   <- R2(nn_predictions_caret, test_data$Concrete_Compressive_Strength)

cat("Modele z pakietów:\n")
cat("KNN RMSE:", caret_knn_rmse, "R^2:", caret_knn_r2, "\n")
cat("Drzewo Decyzyjne RMSE:", caret_tree_rmse, "R^2:", caret_tree_r2, "\n")
cat("Sieć Neuronowa RMSE:", caret_nn_rmse, "R^2:", caret_nn_r2, "\n")

# Łączymy wyniki dla RMSE i R^2
results_comparison <- data.frame(
  Model = rep(c("KNN", "Drzewo Decyzyjne", "Sieć Neuronowa"), 2),
  Podejście = rep(c("Własne", "Pakietowe"), each = 3),
  RMSE = c(knn_rmse, tree_rmse, nn_rmse, caret_knn_rmse, caret_tree_rmse, caret_nn_rmse),
  R2   = c(knn_r2, tree_r2, nn_r2, caret_knn_r2, caret_tree_r2, caret_nn_r2)
)

# Wykres RMSE
ggplot(results_comparison, aes(x = Model, y = RMSE, fill = Podejście)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Porównanie RMSE: Własne vs Pakietowe") +
  theme_minimal()

# Wykres R^2
ggplot(results_comparison, aes(x = Model, y = R2, fill = Podejście)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Porównanie R^2: Własne vs Pakietowe") +
  theme_minimal()

# ============================================================
# Problem Klasyfikacji binarnej
# ============================================================

source("funkcje.R")

# Wczytanie danych
data <- read.csv("bank.csv", sep = ";")

# Konwersja zmiennej celu na 0/1
data$y <- ifelse(data$y == "yes", 1, 0)

# Lista zmiennych kategorycznych
categorical_cols <- c("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome")

for (col in categorical_cols) {
  data[[col]] <- as.factor(data[[col]])
}

# One-hot encoding
encoded_data <- model.matrix(y ~ . - 1, data = data)
encoded_data <- as.data.frame(encoded_data)
encoded_data$y <- data$y

# Standaryzacja
features_scaled <- as.data.frame(scale(encoded_data[, -ncol(encoded_data)]))
target <- encoded_data$y

# Podział na zbiór treningowy i testowy (80/20)
set.seed(123)
train_indices <- sample(1:nrow(features_scaled), 0.8 * nrow(features_scaled))
train_X <- features_scaled[train_indices, ]
train_y <- target[train_indices]
test_X <- features_scaled[-train_indices, ]
test_y <- target[-train_indices]

# Parametry modeli
k <- 10  # l. sąsiadów dla KNN
max_depth <- 5  # max głębokość drzewa
hidden_neurons <- 50  # l. neuronów ukrytych
epochs <- 2000  # l. epok
learning_rate <- 0.0005  # wsp. uczenia
threshold <- 0.5  # próg decyzyjny dla modeli

# trening modeli
knn_model <- train_knn_classification(train_X, train_y)
tree_model <- train_tree_classification(train_X, train_y, max_depth)
nn_model <- train_nn_classification(train_X, train_y, hidden_neurons, epochs, learning_rate)

# predykcja modeli
knn_predictions <- predict_knn_classification(knn_model, test_X, k, threshold)
tree_predictions <- predict_tree_classification(tree_model, test_X)
nn_predictions <- predict_nn_classification(nn_model, test_X)

# Funkcja do obliczania dokłądności
accuracy <- function(actual, predicted) {
  sum(actual == predicted) / length(actual)
}

# Obliczanie dokładności dla każdego modelu
knn_acc <- accuracy(test_y, knn_predictions)
tree_acc <- accuracy(test_y, tree_predictions)
nn_acc <- accuracy(test_y, nn_predictions)

cat("KNN Accuracy:", knn_acc, "\n")
cat("Drzewo Decyzyjne Accuracy:", tree_acc, "\n")
cat("Sieć Neuronowa Accuracy:", nn_acc, "\n")

# Wyniki
results <- data.frame(
  Model = c("KNN", "Drzewo Decyzyjne", "Sieć Neuronowa"),
  Accuracy = c(knn_acc, tree_acc, nn_acc)
)

# Wykres dokładności modeli
ggplot(results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", color = "black") +
  ggtitle("Porównanie dokładności modeli klasyfikacyjnych") +
  theme_minimal()

# Macierz pomyłek
confusion_matrix <- function(actual, predicted) {
  table(Predicted = predicted, Actual = actual)
}

plot_confusion_matrix <- function(actual, predicted, model_name) {
  cm <- as.data.frame(table(Predicted = predicted, Actual = actual))
  
  ggplot(cm, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "black") +
    scale_fill_gradient(low = "white", high = "red") +
    geom_text(aes(label = Freq), vjust = 1.5, color = "black", size = 5) +
    ggtitle(paste("Macierz Pomyłek -", model_name)) +
    theme_minimal()
}

# Macierzy pomyłek
plot_confusion_matrix(test_y, knn_predictions, "KNN")
plot_confusion_matrix(test_y, tree_predictions, "Drzewo Decyzyjne")
plot_confusion_matrix(test_y, nn_predictions, "Sieć Neuronowa")

train_data_cls <- cbind(train_X, y = as.factor(train_y))
test_data_cls  <- cbind(test_X,  y = as.factor(test_y))

levels(train_data_cls$y) <- make.names(levels(train_data_cls$y))
levels(test_data_cls$y)  <- make.names(levels(test_data_cls$y))

# Model KNN przy użyciu caret
set.seed(123)
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)
knn_model_pkg <- train(y ~ ., 
                       data = train_data_cls, 
                       method = "knn", 
                       tuneGrid = data.frame(k = 10),
                       trControl = ctrl)
knn_predictions_pkg <- predict(knn_model_pkg, newdata = test_data_cls)
knn_acc_pkg <- mean(knn_predictions_pkg == test_data_cls$y)

# Drzewo decyzyjne przy użyciu rpart
tree_model_pkg <- rpart::rpart(y ~ ., 
                               data = train_data_cls, 
                               method = "class", 
                               control = rpart.control(maxdepth = 5))
tree_predictions_pkg <- predict(tree_model_pkg, newdata = test_data_cls, type = "class")
tree_acc_pkg <- mean(tree_predictions_pkg == test_data_cls$y)

# Sieć neuronowa przy użyciu nnet
hidden_neurons <- 50
epochs <- 2000
learning_rate <- 0.0005
threshold <- 0.5

x_train <- as.matrix(train_data_cls[, -ncol(train_data_cls)])
y_train <- class.ind(train_data_cls$y)

nn_model_pkg <- nnet::nnet(x_train, y_train,
                           size = hidden_neurons, 
                           maxit = epochs, 
                           decay = learning_rate, 
                           trace = FALSE,
                           MaxNWts = 10000)

x_test <- as.matrix(test_data_cls[, -ncol(test_data_cls)])
nn_probs_pkg <- predict(nn_model_pkg, newdata = x_test, type = "raw")

if (is.matrix(nn_probs_pkg) && ncol(nn_probs_pkg) > 1) {
  nn_pred_numeric <- ifelse(nn_probs_pkg[,2] > threshold, 1, 0)
} else {
  nn_pred_numeric <- ifelse(nn_probs_pkg > threshold, 1, 0)
}

nn_predictions_pkg <- factor(nn_pred_numeric, levels = c(0, 1), labels = levels(test_data_cls$y))
nn_acc_pkg <- mean(nn_predictions_pkg == test_data_cls$y)

# Wyświetlenie wyników i macierzy pomyłek
cat("Pakietowe modele:\n")
cat("KNN Accuracy:", knn_acc_pkg, "\n")
cat("Drzewo Decyzyjne Accuracy:", tree_acc_pkg, "\n")
cat("Sieć Neuronowa Accuracy:", nn_acc_pkg, "\n")

results_pkg <- data.frame(
  Model = c("KNN", "Drzewo Decyzyjne", "Sieć Neuronowa"),
  Accuracy = c(knn_acc_pkg, tree_acc_pkg, nn_acc_pkg)
)

ggplot(results_pkg, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Dokładność modeli pakietowych") +
  theme_minimal()

cat("KNN Confusion Matrix:\n")
print(table(Predicted = knn_predictions_pkg, Actual = test_data_cls$y))

cat("Drzewo Decyzyjne Confusion Matrix:\n")
print(table(Predicted = tree_predictions_pkg, Actual = test_data_cls$y))

cat("Sieć Neuronowa Confusion Matrix:\n")
print(table(Predicted = nn_predictions_pkg, Actual = test_data_cls$y))

results_comparison <- data.frame(
  Model = rep(c("KNN", "Drzewo Decyzyjne", "Sieć Neuronowa"), 2),
  Podejście = rep(c("Własne", "Pakietowe"), each = 3),
  Accuracy = c(knn_acc, tree_acc, nn_acc, knn_acc_pkg, tree_acc_pkg, nn_acc_pkg)
)

print(results_comparison)

ggplot(results_comparison, aes(x = Model, y = Accuracy, fill = Podejście)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  ggtitle("Porównanie dokładności modeli: Własne vs. Pakietowe") +
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

