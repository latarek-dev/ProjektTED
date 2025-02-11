

# Funkcja do konwersji kolumn nieliczbowych do wartości liczbowych
safe_numeric_conversion <- function(column) {
  if (is.numeric(column)) {
    return(column)
  } else {
    cat("Unikalne wartości w kolumnie:\n", unique(column), "\n")
    converted <- suppressWarnings(as.numeric(as.character(column)))
    if (any(is.na(converted) & !is.na(column))) {
      problematic_values <- unique(column[is.na(converted) & !is.na(column)])
      cat("Wartości, które spowodowały NA podczas konwersji:\n", problematic_values, "\n")
      stop("Konwersja na liczby wprowadziła wartości NA. Sprawdź dane wejściowe.")
    }
    return(converted)
  }
}


# --- KNN (regresja) ---
knn_my_regression <- function(train_data, train_labels, test_data, k = 3) {
  predictions <- apply(test_data, 1, function(test_row) {
    distances <- sqrt(rowSums((train_data - test_row)^2))
    neighbors <- order(distances)[1:k]
    neighbor_labels <- train_labels[neighbors]
    predicted_value <- mean(neighbor_labels, na.rm = TRUE)
    return(predicted_value)
  })
  return(predictions)
}

# Funkcja do obliczenia błędu MSE
mse <- function(predictions, true_values) {
  return(mean((predictions - true_values)^2, na.rm = TRUE))
}

# --- Drzewo decyzyjne (regresja) ---
decision_tree_reg <- function(data, labels, depth = 3) {
  # Warunek zakończenia: głębokość 0 lub wszystkie etykiety są takie same
  if (depth == 0 || length(unique(labels)) == 1) {
    return(list(value = mean(labels, na.rm = TRUE), feature = NULL))
  }
  
  best_split <- NULL
  best_mse <- Inf
  
  # Przegląd wszystkich cech
  for (feature in 1:ncol(data)) {
    # Ustal punkt podziału jako średnią wartość danej cechy
    split_value <- mean(data[, feature], na.rm = TRUE)
    left_idx <- data[, feature] <= split_value
    right_idx <- data[, feature] > split_value
    
    # Jeśli jeden z podziałów jest pusty, pomiń tę cechę
    if (sum(left_idx) == 0 || sum(right_idx) == 0) {
      next
    }
    
    left <- labels[left_idx]
    right <- labels[right_idx]
    
    # Oblicz MSE dla lewego i prawego podziału
    mse_left <- mean((left - mean(left, na.rm = TRUE))^2, na.rm = TRUE)
    mse_right <- mean((right - mean(right, na.rm = TRUE))^2, na.rm = TRUE)
    
    # Łączne MSE jako średnia ważona
    mse_val <- (sum(left_idx) / length(labels)) * mse_left + 
      (sum(right_idx) / length(labels)) * mse_right
    
    if (!is.na(mse_val) && mse_val < best_mse) {
      best_mse <- mse_val
      best_split <- list(feature = feature, value = split_value)
    }
  }
  
  # Jeśli nie znaleziono dobrego podziału, zwróć liść
  if (is.null(best_split)) {
    return(list(value = mean(labels, na.rm = TRUE), feature = NULL))
  }
  
  # Podział danych według najlepszego kryterium
  left_idx <- data[, best_split$feature] <= best_split$value
  right_idx <- data[, best_split$feature] > best_split$value
  
  left_tree <- decision_tree_reg(data[left_idx, , drop = FALSE], labels[left_idx], depth - 1)
  right_tree <- decision_tree_reg(data[right_idx, , drop = FALSE], labels[right_idx], depth - 1)
  
  return(list(
    feature = best_split$feature,
    value = best_split$value,
    left = left_tree,
    right = right_tree
  ))
}

# Funkcja predykcji dla drzewa decyzyjnego
predict_tree_reg <- function(tree, data) {
  # Jeśli pole 'feature' jest NULL, mamy do czynienia z liściem
  if (is.null(tree$feature)) {
    return(rep(tree$value, nrow(data)))
  }
  
  # Podział danych na podstawie reguły węzła
  left_idx <- data[, tree$feature] <= tree$value
  predictions <- rep(NA, nrow(data))
  
  predictions[left_idx] <- predict_tree_reg(tree$left, data[left_idx, , drop = FALSE])
  predictions[!left_idx] <- predict_tree_reg(tree$right, data[!left_idx, , drop = FALSE])
  
  return(predictions)
}

# --- Sieć neuronowa (regresja) ---
neural_network_my <- function(train_data, train_labels, hidden_layers = c(5), epochs = 100, learning_rate = 0.01) {
  # Konwersja danych do macierzy numerycznych
  train_data <- as.matrix(train_data)
  mode(train_data) <- "numeric"
  train_labels <- as.numeric(train_labels)
  
  if (nrow(train_data) == 0) stop("Błąd: zbiór train_data jest pusty!")
  if (length(train_labels) != nrow(train_data)) stop("Błąd: train_labels i train_data mają różne rozmiary!")
  
  input_neurons <- ncol(train_data)
  output_neurons <- 1  # dla regresji
  num_hidden_layers <- length(hidden_layers)
  
  # Inicjalizacja list na wagi (W) i biasy (b)
  W <- list()
  b <- list()
  
  # Warstwa 1: wejście -> pierwsza warstwa ukryta
  W[[1]] <- matrix(runif(input_neurons * hidden_layers[1], -0.5, 0.5),
                   nrow = input_neurons, ncol = hidden_layers[1])
  b[[1]] <- matrix(runif(hidden_layers[1], -0.5, 0.5),
                   nrow = 1, ncol = hidden_layers[1])
  
  # Inicjalizacja kolejnych warstw ukrytych (jeśli są)
  if (num_hidden_layers > 1) {
    for (l in 2:num_hidden_layers) {
      W[[l]] <- matrix(runif(hidden_layers[l-1] * hidden_layers[l], -0.5, 0.5),
                       nrow = hidden_layers[l-1], ncol = hidden_layers[l])
      b[[l]] <- matrix(runif(hidden_layers[l], -0.5, 0.5),
                       nrow = 1, ncol = hidden_layers[l])
    }
  }
  
  # Warstwa wyjściowa: ostatnia warstwa ukryta -> wyjście
  W_out <- matrix(runif(hidden_layers[num_hidden_layers] * output_neurons, -0.5, 0.5),
                  nrow = hidden_layers[num_hidden_layers], ncol = output_neurons)
  b_out <- matrix(runif(output_neurons, -0.5, 0.5),
                  nrow = 1, ncol = output_neurons)
  
  # Trening – pętla po epokach
  for (epoch in 1:epochs) {
    total_loss <- 0
    for (i in seq_len(nrow(train_data))) {
      x <- matrix(train_data[i, , drop = FALSE], nrow = 1)
      y <- train_labels[i]
      
      activations <- list()
      zs <- list()
      
      # Forward pass: pierwsza warstwa
      zs[[1]] <- x %*% W[[1]] + b[[1]]
      activations[[1]] <- tanh(zs[[1]])
      
      if (num_hidden_layers > 1) {
        for (l in 2:num_hidden_layers) {
          zs[[l]] <- activations[[l-1]] %*% W[[l]] + b[[l]]
          activations[[l]] <- tanh(zs[[l]])
        }
      }
      
      # Warstwa wyjściowa
      z_out <- activations[[num_hidden_layers]] %*% W_out + b_out
      
      # Używamy MSE jako funkcji straty
      loss <- (z_out - y)^2
      total_loss <- total_loss + loss
      
      # Backpropagation – gradient funkcji MSE
      delta_out <- 2 * (z_out - y)
      grad_W_out <- t(activations[[num_hidden_layers]]) %*% delta_out
      grad_b_out <- delta_out
      
      delta <- (delta_out %*% t(W_out)) * (1 - tanh(zs[[num_hidden_layers]])^2)
      grad_W <- vector("list", num_hidden_layers)
      grad_b <- vector("list", num_hidden_layers)
      
      if (num_hidden_layers == 1) {
        prev_activation <- x
      } else {
        prev_activation <- activations[[num_hidden_layers - 1]]
      }
      grad_W[[num_hidden_layers]] <- t(prev_activation) %*% delta
      grad_b[[num_hidden_layers]] <- delta
      
      if (num_hidden_layers > 1) {
        for (l in seq(num_hidden_layers - 1, 1)) {
          if (l == 1) {
            prev_activation <- x
          } else {
            prev_activation <- activations[[l-1]]
          }
          grad_W[[l]] <- t(prev_activation) %*% delta
          grad_b[[l]] <- delta
          if (l > 1) {
            delta <- (delta %*% t(W[[l]])) * (1 - tanh(zs[[l-1]])^2)
          }
        }
      }
      
      W_out <- W_out - learning_rate * grad_W_out
      b_out <- b_out - learning_rate * grad_b_out
      
      for (l in 1:num_hidden_layers) {
        W[[l]] <- W[[l]] - learning_rate * grad_W[[l]]
        b[[l]] <- b[[l]] - learning_rate * grad_b[[l]]
      }
    }
    cat("Epoch:", epoch, "MSE Loss:", total_loss / nrow(train_data), "\n")
  }
  
  return(list(W = W, b = b, W_out = W_out, b_out = b_out, hidden_layers = hidden_layers))
}

# Funkcja predykcji sieci neuronowej
predict_nn_reg <- function(model, test_data) {
  # Załóż, że dane są już przeskalowane – więc nie wykonujemy dodatkowej normalizacji
  test_data <- as.matrix(test_data)
  
  # Ustal liczbę neuronów wejściowych na podstawie wymiarów macierzy wag z pierwszej warstwy
  input_neurons <- nrow(model$W[[1]])
  
  if (ncol(test_data) != input_neurons) {
    stop(paste("BŁĄD: Niezgodne wymiary! test_data:", ncol(test_data),
               "!= oczekiwane:", input_neurons))
  }
  
  # Poprawne określenie liczby warstw ukrytych – używamy pola 'hidden_layers'
  num_hidden_layers <- length(model$hidden_layers)
  activations <- test_data
  
  # Przepływ przez warstwy ukryte – stosujemy tę samą funkcję aktywacji, co podczas treningu (tanh)
  for (l in 1:num_hidden_layers) {
    activations <- tanh(activations %*% model$W[[l]] +
                          matrix(rep(model$b[[l]], nrow(activations)),
                                 ncol = ncol(model$W[[l]]), byrow = TRUE))
  }
  
  # Warstwa wyjściowa – liniowa kombinacja (bez aktywacji, odpowiednia dla regresji)
  z_out <- activations %*% model$W_out +
    matrix(rep(model$b_out, nrow(activations)),
           ncol = ncol(model$W_out), byrow = TRUE)
  
  # Zwracamy predykcję jako wartość numeryczną
  return(as.numeric(z_out))
}

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
  return(mse_fold)  
}

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
  return(mse_fold)
}

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



#klasYFIKACJA_BINARNA##

# Implementacja KNN dla klasyfikacji binarnej
knn_binary <- function(train_data, train_labels, test_data, k = 3) {
  predictions <- sapply(1:nrow(test_data), function(i) {
    distances <- sqrt(rowSums((train_data - test_data[i, ])^2))
    neighbors <- order(distances)[1:k]
    predicted_label <- as.numeric(names(which.max(table(train_labels[neighbors]))))
    return(predicted_label)
  })
  return(predictions)
}

#drzewo_dec

find_best_split <- function(data, labels) {
  best_split <- NULL
  best_entropy <- Inf
  
  for (feature in 1:ncol(data)) {
    unique_values <- unique(data[, feature])
    
    for (split_value in unique_values) {
      left <- labels[data[, feature] <= split_value]
      right <- labels[data[, feature] > split_value]
      
      if (length(left) == 0 || length(right) == 0) next  
      
      #Entropia
      entropy_left <- -sum((table(left) / length(left)) * log2(table(left) / length(left) + 1e-10))
      entropy_right <- -sum((table(right) / length(right)) * log2(table(right) / length(right) + 1e-10))
      entropy <- (length(left) * entropy_left + length(right) * entropy_right) / length(labels)
      
      if (entropy < best_entropy) {
        best_entropy <- entropy
        best_split <- list(feature = feature, value = split_value)
      }
    }
  }
  
  return(best_split)
}

# Funkcja drzewa decyzyjnego
decision_tree_binary <- function(data, labels, depth = 3) {
  # zakończenie f.: głębokość 0 lub wszystkie etykiety są takie same
  if (depth == 0 || length(unique(labels)) == 1) {
    return(list(value = as.numeric(names(which.max(table(labels)))))) 
  }
  
  best_split <- find_best_split(data, labels)
  
  if (is.null(best_split)) {
    return(list(value = as.numeric(names(which.max(table(labels)))))) 
  }
  
  left_idx <- data[, best_split$feature] <= best_split$value
  right_idx <- !left_idx
  
  return(list(
    feature = best_split$feature,
    value = best_split$value,
    left = decision_tree_binary(data[left_idx, , drop = FALSE], labels[left_idx], depth - 1),
    right = decision_tree_binary(data[right_idx, , drop = FALSE], labels[right_idx], depth - 1)
  ))
}

# Predykcja na drzewa
predict_tree_binary <- function(tree, data) {
  if (is.null(tree$feature)) {
    return(rep(tree$value, nrow(data)))
  }
  left_idx <- data[, tree$feature] <= tree$value
  predictions <- rep(NA, nrow(data))
  predictions[left_idx] <- predict_tree_binary(tree$left, data[left_idx, , drop = FALSE])
  predictions[!left_idx] <- predict_tree_binary(tree$right, data[!left_idx, , drop = FALSE])
  return(predictions)
}

# Funkcja do obliczania dokładności
accuracy_b <- function(predictions, true_labels) {
  mean(predictions == true_labels)
}

# Implementacja sieci neuronowej dla klas. binarnej
neural_network_binary <- function(train_data, train_labels, hidden_neurons = c(5), epochs = 50, learning_rate = 0.001) {
  # Konwersja danych do macierzy numerycznych
  train_data <- as.matrix(train_data)
  mode(train_data) <- "numeric"
  train_labels <- as.numeric(train_labels)
  
  input_neurons <- ncol(train_data)
  output_neurons <- 1
  
  num_hidden_layers <- length(hidden_neurons)
  
  # Inicjalizacja list na wagi i biasy dla warstw ukrytych
  W <- list()
  b <- list()
  
  # Pierwsza warstwa ukryta: wejście -> pierwsza warstwa
  W[[1]] <- matrix(runif(input_neurons * hidden_neurons[1], -0.5, 0.5),
                   nrow = input_neurons, ncol = hidden_neurons[1])
  b[[1]] <- runif(hidden_neurons[1], -0.5, 0.5)
  
  # Kolejne warstwy ukryte, jeśli są
  if(num_hidden_layers > 1) {
    for(l in 2:num_hidden_layers) {
      W[[l]] <- matrix(runif(hidden_neurons[l-1] * hidden_neurons[l], -0.5, 0.5),
                       nrow = hidden_neurons[l-1], ncol = hidden_neurons[l])
      b[[l]] <- runif(hidden_neurons[l], -0.5, 0.5)
    }
  }
  
  # Warstwa wyjściowa: ostatnia warstwa ukryta -> wyjście
  W_out <- matrix(runif(hidden_neurons[num_hidden_layers] * output_neurons, -0.5, 0.5),
                  nrow = hidden_neurons[num_hidden_layers], ncol = output_neurons)
  b_out <- runif(output_neurons, -0.5, 0.5)
  
  # Trening – pętla po epokach
  for (epoch in 1:epochs) {
    for (i in 1:nrow(train_data)) {
      x <- matrix(train_data[i, ], nrow = 1)
      y <- train_labels[i]
      
      # Forward pass
      activations <- list()
      zs <- list()
      
      # Pierwsza warstwa ukryta
      zs[[1]] <- x %*% W[[1]] + b[[1]]
      activations[[1]] <- tanh(zs[[1]])
      
      # Kolejne warstwy ukryte
      if(num_hidden_layers > 1) {
        for (l in 2:num_hidden_layers) {
          zs[[l]] <- activations[[l-1]] %*% W[[l]] + b[[l]]
          activations[[l]] <- tanh(zs[[l]])
        }
      }
      
      # Warstwa wyjściowa – używamy funkcji sigmoid
      z_out <- activations[[num_hidden_layers]] %*% W_out + b_out
      a2 <- 1 / (1 + exp(-z_out))
      
      # Obliczenie błędu przy użyciu MSE
      error <- a2 - y
      delta_out <- 2 * error * a2 * (1 - a2)
      
      # Backpropagation dla warstwy wyjściowej
      grad_W_out <- t(activations[[num_hidden_layers]]) %*% delta_out
      grad_b_out <- delta_out
      
      # Propagacja błędu do ostatniej warstwy ukrytej
      delta <- (delta_out %*% t(W_out)) * (1 - tanh(zs[[num_hidden_layers]])^2)
      
      # Inicjalizacja listy gradientów dla warstw ukrytych
      grad_W <- vector("list", num_hidden_layers)
      grad_b <- vector("list", num_hidden_layers)
      
      if(num_hidden_layers == 1) {
        prev_activation <- x
      } else {
        prev_activation <- activations[[num_hidden_layers - 1]]
      }
      grad_W[[num_hidden_layers]] <- t(prev_activation) %*% delta
      grad_b[[num_hidden_layers]] <- delta
      
      # Backpropagation przez pozostałe warstwy ukryte, jeśli istnieją
      if(num_hidden_layers > 1) {
        for (l in seq(num_hidden_layers - 1, 1)) {
          if (l == 1) {
            prev_activation <- x
          } else {
            prev_activation <- activations[[l-1]]
          }
          grad_W[[l]] <- t(prev_activation) %*% delta
          grad_b[[l]] <- delta
          if (l > 1) {
            delta <- (delta %*% t(W[[l]])) * (1 - tanh(zs[[l-1]])^2)
          }
        }
      }
      
      # Aktualizacja wag i biasów
      W_out <- W_out - learning_rate * grad_W_out
      b_out <- b_out - learning_rate * grad_b_out
      
      for (l in 1:num_hidden_layers) {
        W[[l]] <- W[[l]] - learning_rate * grad_W[[l]]
        b[[l]] <- b[[l]] - learning_rate * grad_b[[l]]
      }
    }
    cat("Epoch:", epoch, "\n")
  }
  
  return(list(W = W, b = b, W_out = W_out, b_out = b_out, hidden_neurons = hidden_neurons))
}

# Funkcja predykcji dla sieci neuronowej dla klasyfikacji binarnej
predict_nn_binary <- function(model, test_data) {
  test_data <- as.matrix(test_data)
  
  if (ncol(test_data) != nrow(model$W[[1]])) {
    stop(paste("Liczba cech w danych testowych (", ncol(test_data),
               ") nie zgadza się z liczbą neuronów wejściowych (",
               nrow(model$W[[1]]), ")!"))
  }
  
  activations <- test_data
  num_hidden_layers <- length(model$hidden_neurons)
  
  for (l in 1:num_hidden_layers) {
    activations <- tanh(activations %*% model$W[[l]] +
                          matrix(rep(model$b[[l]], nrow(activations)),
                                 ncol = length(model$b[[l]]), byrow = TRUE))
  }
  
  z_out <- activations %*% model$W_out +
    matrix(rep(model$b_out, nrow(activations)),
           ncol = length(model$b_out), byrow = TRUE)
  a2 <- 1 / (1 + exp(-z_out))
  
  predictions <- ifelse(a2 > 0.5, 1, 0)
  return(predictions)
}
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



































#klas.wieloklasowa
library(class)  

#Implementacja KNN dla klas. wieloklasowej 
euclidean_distance <- function(x1, x2) {
  sqrt(sum((x1 - x2)^2))
}

knn_wieloklasowa <- function(train_data, train_labels, test_data, k) {
  predictions <- numeric(nrow(test_data))
  
  for (i in 1:nrow(test_data)) {
    distances <- apply(train_data, 1, euclidean_distance, x2 = test_data[i, ])
    neighbors_idx <- order(distances)[1:k]
    nearest_labels <- train_labels[neighbors_idx]
    predictions[i] <- as.numeric(names(which.max(table(nearest_labels))))
  }
  
  return(predictions)
}





sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

sigmoid_derivative <- function(x) {
  x * (1 - x)
}

normalize_data <- function(data) {
  return((data - min(data)) / (max(data) - min(data) + 1e-9))  
}
#Funkcja sieci neuronowych 
neural_network_train <- function(train_data, train_labels, hidden_neurons = 10, epochs = 200, learning_rate = 0.01) {
  # Jeśli hidden_neurons nie jest wektorem, zamień na wektor
  if (length(hidden_neurons) == 1) {
    hidden_neurons <- c(hidden_neurons)
  }
  
  train_data <- normalize_data(train_data)  
  
  input_neurons <- ncol(train_data)
  output_neurons <- length(unique(train_labels))
  
  # One-hot encoding dla etykiet
  train_labels_encoded <- matrix(0, nrow = length(train_labels), ncol = output_neurons)
  for (i in 1:length(train_labels)) {
    train_labels_encoded[i, train_labels[i]] <- 1
  }
  
  num_hidden_layers <- length(hidden_neurons)
  
  # Inicjalizacja wag i biasów dla warstw ukrytych
  W <- list()
  b <- list()
  
  # Warstwa 1: wejście -> pierwsza warstwa ukryta
  W[[1]] <- matrix(runif(input_neurons * hidden_neurons[1], -0.5, 0.5),
                   nrow = input_neurons, ncol = hidden_neurons[1])
  b[[1]] <- matrix(runif(hidden_neurons[1], -0.5, 0.5),
                   nrow = 1, ncol = hidden_neurons[1])
  
  # Kolejne warstwy ukryte (jeśli są)
  if (num_hidden_layers > 1) {
    for (l in 2:num_hidden_layers) {
      W[[l]] <- matrix(runif(hidden_neurons[l-1] * hidden_neurons[l], -0.5, 0.5),
                       nrow = hidden_neurons[l-1], ncol = hidden_neurons[l])
      b[[l]] <- matrix(runif(hidden_neurons[l], -0.5, 0.5),
                       nrow = 1, ncol = hidden_neurons[l])
    }
  }
  
  # Warstwa wyjściowa: ostatnia warstwa ukryta -> wyjście
  W_out <- matrix(runif(hidden_neurons[num_hidden_layers] * output_neurons, -0.5, 0.5),
                  nrow = hidden_neurons[num_hidden_layers], ncol = output_neurons)
  b_out <- matrix(runif(output_neurons, -0.5, 0.5),
                  nrow = 1, ncol = output_neurons)
  
  for (epoch in 1:epochs) {
    total_loss <- 0
    
    for (i in 1:nrow(train_data)) {
      x <- matrix(train_data[i, , drop = FALSE], nrow = 1)  
      y <- train_labels_encoded[i, , drop = FALSE]
      
      # Forward pass
      activations <- list()
      zs <- list()
      
      # Warstwa 1
      zs[[1]] <- x %*% W[[1]] + b[[1]]
      activations[[1]] <- sigmoid(zs[[1]])
      
      # Kolejne warstwy ukryte
      if(num_hidden_layers > 1) {
        for(l in 2:num_hidden_layers) {
          zs[[l]] <- activations[[l-1]] %*% W[[l]] + b[[l]]
          activations[[l]] <- sigmoid(zs[[l]])
        }
      }
      
      # Warstwa wyjściowa
      z_out <- activations[[num_hidden_layers]] %*% W_out + b_out
      a_out <- sigmoid(z_out)
      
      loss <- sum((a_out - y)^2)
      total_loss <- total_loss + loss
      
      # Backpropagation dla wyjścia
      delta_out <- 2 * (a_out - y) * sigmoid_derivative(a_out)
      grad_W_out <- t(activations[[num_hidden_layers]]) %*% delta_out
      grad_b_out <- delta_out
      
      # Backpropagation dla ostatniej warstwy ukrytej
      delta <- (delta_out %*% t(W_out)) * sigmoid_derivative(activations[[num_hidden_layers]])
      
      grad_W <- vector("list", num_hidden_layers)
      grad_b <- vector("list", num_hidden_layers)
      
      # Dla ostatniej warstwy ukrytej
      if (num_hidden_layers == 1) {
        prev_activation <- x
      } else {
        prev_activation <- activations[[num_hidden_layers - 1]]
      }
      grad_W[[num_hidden_layers]] <- t(prev_activation) %*% delta
      grad_b[[num_hidden_layers]] <- delta
      
      # Backpropagation przez pozostałe warstwy
      if(num_hidden_layers > 1) {
        for(l in seq(num_hidden_layers - 1, 1)) {
          if(l == 1) {
            prev_activation <- x
          } else {
            prev_activation <- activations[[l-1]]
          }
          grad_W[[l]] <- t(prev_activation) %*% delta
          grad_b[[l]] <- delta
          if(l > 1) {
            delta <- (delta %*% t(W[[l]])) * sigmoid_derivative(activations[[l-1]])
          }
        }
      }
      
      # Aktualizacja wag
      W_out <- W_out - learning_rate * grad_W_out
      b_out <- b_out - learning_rate * grad_b_out
      
      for(l in 1:num_hidden_layers) {
        W[[l]] <- W[[l]] - learning_rate * grad_W[[l]]
        b[[l]] <- b[[l]] - learning_rate * grad_b[[l]]
      }
    }
    
    if (epoch %% 10 == 0) {
      cat("Epoka:", epoch, "Błąd:", total_loss / nrow(train_data), "\n")
    }
  }
  
  return(list(W = W, b = b, W_out = W_out, b_out = b_out, 
              input_neurons = input_neurons, hidden_neurons = hidden_neurons))
}

# Funkcja predykcji dla sieci neuronowej (klasyfikacja wieloklasowa)
predict_nn <- function(model, test_data) {
  test_data <- normalize_data(test_data)
  test_data <- as.matrix(test_data)
  
  if (ncol(test_data) != model$input_neurons) {
    stop(paste("BŁĄD: Niezgodne wymiary! test_data:", ncol(test_data),
               "!= model$input_neurons:", model$input_neurons))
  }
  
  num_hidden_layers <- length(model$hidden_neurons)
  activations <- test_data
  
  for(l in 1:num_hidden_layers) {
    activations <- sigmoid(activations %*% model$W[[l]] + 
                             matrix(rep(model$b[[l]], nrow(activations)),
                                    ncol = length(model$b[[l]]), byrow = TRUE))
  }
  
  z_out <- activations %*% model$W_out + 
    matrix(rep(model$b_out, nrow(activations)),
           ncol = length(model$b_out), byrow = TRUE)
  a_out <- sigmoid(z_out)
  
  return(max.col(a_out))
}


#F. dokładności
accuracy <- function(predictions, true_labels) {
  mean(predictions == true_labels, na.rm = TRUE)
}


#drzewa decyzyjne
decision_tree_w <- function(data, labels, depth = 5, criterion = "gini") {
  if (depth == 0 || length(unique(labels)) == 1 || nrow(data) == 0) {
    return(list(value = as.character(names(which.max(table(labels))))))  
  }
  
  best_split <- NULL
  best_score <- Inf
  
  gini_index <- function(labels) {
    proportions <- table(labels) / length(labels)
    return(1 - sum(proportions^2))
  }
  
  entropy_index <- function(labels) {
    proportions <- table(labels) / length(labels)
    return(-sum(proportions * log2(proportions + 1e-9)))  
  }
  
  impurity_function <- ifelse(criterion == "entropy", entropy_index, gini_index)
  
  for (feature in 1:ncol(data)) {
    split_value <- quantile(data[, feature], 0.5, na.rm = TRUE)  
    
    left_idx <- data[, feature] <= split_value
    right_idx <- !left_idx
    
    left_labels <- labels[left_idx]
    right_labels <- labels[right_idx]
    
    if (length(left_labels) == 0 || length(right_labels) == 0) next  
    
    impurity_left <- impurity_function(left_labels)
    impurity_right <- impurity_function(right_labels)
    score <- (length(left_labels) / length(labels)) * impurity_left + 
      (length(right_labels) / length(labels)) * impurity_right
    
    if (score < best_score) {
      best_score <- score
      best_split <- list(feature = feature, value = split_value)
    }
  }
  
  if (is.null(best_split)) {
    return(list(value = as.character(names(which.max(table(labels))))))  
  }
  
  left_idx <- data[, best_split$feature] <= best_split$value
  right_idx <- !left_idx
  
  return(list(
    feature = best_split$feature,
    value = best_split$value,
    left = decision_tree_w(data[left_idx, , drop = FALSE], labels[left_idx], depth - 1, criterion),
    right = decision_tree_w(data[right_idx, , drop = FALSE], labels[right_idx], depth - 1, criterion)
  ))
}

#Funkcja predykcji dla drzewa
predict_tree_w <- function(tree, data) {
  predictions <- character(nrow(data))
  
  for (i in 1:nrow(data)) {
    node <- tree
    while (!is.null(node$feature)) {
      if (!is.null(node$left) && !is.null(node$right)) {
        if (data[i, node$feature] <= node$value) {
          node <- node$left
        } else {
          node <- node$right
        }
      } else {
        break
      }
    }
    if (!is.null(node$value)) {
      predictions[i] <- node$value
    } else {
      predictions[i] <- NA  
    }
  }
  return(predictions)
}

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
