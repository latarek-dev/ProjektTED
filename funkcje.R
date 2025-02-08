# ============================================================
# Funkcje pomocnicze
# ============================================================

normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

r2_score <- function(actual, predicted) {
  ss_total <- sum((actual - mean(actual))^2)
  ss_residual <- sum((actual - predicted)^2)
  1 - (ss_residual / ss_total)
}

accuracy <- function(actual, predicted) {
  sum(actual == predicted) / length(actual)
}

confusion_matrix <- function(actual, predicted) {
  table(Predicted = predicted, Actual = actual)
}

# ===============================
# Funkcja walidacji krzyżowej zwracająca wiele miar
# ===============================
custom_cv <- function(data, k_folds, train_func, predict_func, 
                            performance_funcs,  # np. list(accuracy = accuracy)
                            hyperparams = NULL, target_col, pass_to_predict = TRUE) {
  n <- nrow(data)
  indices <- sample(1:n)  # losowe mieszanie indeksów
  fold_size <- floor(n / k_folds)
  
  # Przygotowujemy listę na wyniki dla każdej miary
  results <- list()
  for (metric in names(performance_funcs)) {
    results[[metric]] <- numeric(k_folds)
  }
  
  for (i in 1:k_folds) {
    if (i < k_folds) {
      test_idx <- indices[((i - 1) * fold_size + 1):(i * fold_size)]
    } else {
      test_idx <- indices[((i - 1) * fold_size + 1):n]
    }
    
    train_data <- data[-test_idx, ]
    test_data  <- data[test_idx, ]
    
    # Rozdzielamy cechy i zmienną celu
    train_X <- train_data[, setdiff(names(train_data), target_col)]
    train_y <- train_data[[target_col]]
    test_X  <- test_data[, setdiff(names(test_data), target_col)]
    test_y  <- test_data[[target_col]]
    
    # Przygotowanie argumentów dla funkcji treningowej
    args_train <- list(train_X, train_y)
    if (!is.null(hyperparams)) {
      args_train <- c(args_train, hyperparams)
    }
    model <- do.call(train_func, args_train)
    
    # Przygotowanie argumentów dla funkcji predykcyjnej
    args_predict <- list(model, test_X)
    if (!is.null(hyperparams) && pass_to_predict) {
      args_predict <- c(args_predict, hyperparams)
    }
    predictions <- do.call(predict_func, args_predict)
    
    # Obliczamy i zapisujemy każdą miarę
    for (metric in names(performance_funcs)) {
      results[[metric]][i] <- performance_funcs[[metric]](test_y, predictions)
    }
  }
  
  # Zwracamy średnie wartości dla każdej miary
  sapply(results, mean)
}

# ============================================================
# Problem Regresji
# ============================================================

# --- Model KNN ---
train_knn <- function(train_X, train_y, ...) {
  args <- list(...)
  if (is.null(args$k)) stop("Podaj wartość k!")
  k <- args$k
  model <- list(X = train_X, y = train_y, k = k)
  return(model)
}

predict_knn <- function(model, test_X, ...) {
  args <- list(...)
  # Używamy przekazanego k, jeśli jest, lub zapisanego w modelu
  k_used <- if (!is.null(args$k)) args$k else model$k
  train_X <- as.matrix(model$X)
  train_y <- model$y
  test_X <- as.matrix(test_X)
  
  n_test <- nrow(test_X)
  predictions <- numeric(n_test)
  
  for (i in 1:n_test) {
    distances <- sqrt(rowSums((train_X - matrix(test_X[i, ], nrow = nrow(train_X),
                                                ncol = ncol(train_X), byrow = TRUE))^2))
    neighbors <- order(distances)[1:k_used]
    predictions[i] <- mean(train_y[neighbors])
  }
  
  return(predictions)
}

# --- Model drzewa decyzyjnego ---
train_tree <- function(train_X, train_y, max_depth, ...) {
  # Funkcja rekurencyjna budująca drzewo
  build_tree <- function(X, y, depth) {
    if (depth == 0 || length(unique(y)) == 1) {
      return(mean(y))
    }
    best_split <- NULL
    best_mse <- Inf
    n <- ncol(X)
    
    for (feature in 1:n) {
      for (split in unique(X[, feature])) {
        left <- y[X[, feature] <= split]
        right <- y[X[, feature] > split]
        if (length(left) == 0 || length(right) == 0) next
        mse <- (sum((left - mean(left))^2) + sum((right - mean(right))^2)) / length(y)
        if (mse < best_mse) {
          best_mse <- mse
          best_split <- list(feature = feature, value = split)
        }
      }
    }
    if (is.null(best_split)) {
      return(mean(y))
    }
    left_indices <- X[, best_split$feature] <= best_split$value
    right_indices <- X[, best_split$feature] > best_split$value
    left_tree <- build_tree(X[left_indices, , drop = FALSE], y[left_indices], depth - 1)
    right_tree <- build_tree(X[right_indices, , drop = FALSE], y[right_indices], depth - 1)
    return(list(split = best_split, left = left_tree, right = right_tree))
  }
  
  tree <- build_tree(as.matrix(train_X), train_y, max_depth)
  model <- list(tree = tree, max_depth = max_depth)
  return(model)
}

predict_tree <- function(model, X, ...) {
  tree <- model$tree
  
  predict_recursive <- function(tree, X_row) {
    if (!is.list(tree)) return(tree)
    if (X_row[tree$split$feature] <= tree$split$value) {
      return(predict_recursive(tree$left, X_row))
    } else {
      return(predict_recursive(tree$right, X_row))
    }
  }
  
  X <- as.matrix(X)
  predictions <- apply(X, 1, function(row) predict_recursive(tree, row))
  return(predictions)
}

# --- Model sieci neuronowej ---
clip_gradient <- function(grad, threshold = 1.0) {
  norm_grad <- sqrt(sum(grad^2))
  if (norm_grad > threshold) {
    grad <- grad * (threshold / norm_grad)
  }
  return(grad)
}

# --- Model sieci neuronowej z gradient clipping ---
train_nn <- function(train_X, train_y, hidden_neurons, epochs, learning_rate, clip_threshold = 1.0, ...) {
  train_X <- as.matrix(train_X)
  train_y <- as.numeric(train_y)
  n_features <- ncol(train_X)
  n_samples <- nrow(train_X)
  
  # Inicjalizacja wag i biasów
  w_hidden <- matrix(rnorm(n_features * hidden_neurons, mean = 0, sd = 0.01), n_features, hidden_neurons)
  b_hidden <- rep(0, hidden_neurons)
  w_output <- matrix(rnorm(hidden_neurons, mean = 0, sd = 0.01), hidden_neurons, 1)
  b_output <- 0
  
  # Funkcja aktywacji: tanh oraz jej pochodna
  tanh_deriv <- function(x) 1 - tanh(x)^2
  
  for (epoch in 1:epochs) {
    hidden_input <- train_X %*% w_hidden + matrix(b_hidden, n_samples, hidden_neurons, byrow = TRUE)
    hidden_output <- tanh(hidden_input)
    output <- hidden_output %*% w_output + b_output
    error <- train_y - output
    loss <- mean(error^2)
    
    if (epoch %% 100 == 0) {
      cat("Epoka:", epoch, "Strata:", loss, "\n")
    }
    
    # Obliczanie gradientów
    d_output <- error  # dla regresji z liniową aktywacją na wyjściu
    d_hidden <- (d_output %*% t(w_output)) * tanh_deriv(hidden_input)
    
    # Zastosowanie gradient clipping
    d_output <- clip_gradient(d_output, threshold = clip_threshold)
    d_hidden <- clip_gradient(d_hidden, threshold = clip_threshold)
    
    # Aktualizacja wag i biasów
    w_output <- w_output + t(hidden_output) %*% d_output * learning_rate
    b_output <- b_output + sum(d_output) * learning_rate
    w_hidden <- w_hidden + t(train_X) %*% d_hidden * learning_rate
    b_hidden <- b_hidden + colSums(d_hidden) * learning_rate
  }
  
  model <- list(w_hidden = w_hidden, b_hidden = b_hidden, w_output = w_output, b_output = b_output)
  return(model)
}

predict_nn <- function(model, test_X, ...) {
  test_X <- as.matrix(test_X)
  hidden_input <- test_X %*% model$w_hidden + matrix(model$b_hidden, nrow(test_X), ncol(model$w_hidden), byrow = TRUE)
  hidden_output <- tanh(hidden_input)
  predictions <- hidden_output %*% model$w_output + model$b_output
  return(as.vector(predictions))
}

# ============================================================
# Klasyfikacja binarna
# ============================================================

# --- k-NN dla klasyfikacji ---
train_knn_classification <- function(train_X, train_y, ...) {
  # Nawet jeśli nie wykorzystujemy hiperparametrów w treningu, dodajemy ...,
  # żeby funkcja mogła przyjmować dodatkowe argumenty przekazywane przez CV.
  return(list(X = train_X, y = train_y))
}

predict_knn_classification <- function(model, test_X, k = 10, threshold = 0.5) {
  train_X <- as.matrix(model$X)
  train_y <- model$y
  test_X <- as.matrix(test_X)
  
  n_test <- nrow(test_X)
  predictions <- numeric(n_test)
  
  for (i in 1:n_test) {
    distances <- sqrt(rowSums((train_X - matrix(test_X[i, ], nrow = nrow(train_X), 
                                                ncol = ncol(train_X), byrow = TRUE))^2))
    neighbors <- order(distances)[1:k]
    # Konwersja etykiet z faktora na liczby
    neighbor_vals <- as.numeric(as.character(train_y[neighbors]))
    # Jeśli średnia z sąsiadów przekracza próg, przypisz 1, w przeciwnym razie 0
    predictions[i] <- ifelse(mean(neighbor_vals) > threshold, 1, 0)
  }
  
  return(predictions)
}


# --- Drzewo decyzyjne dla klasyfikacji ---
train_tree_classification <- function(X, y, depth, ...) {
  if (depth == 0 || length(unique(y)) == 1) {
    return(as.integer(names(sort(table(y), decreasing = TRUE)[1])))
  }
  
  best_split <- NULL
  best_gini <- Inf
  n <- ncol(X)
  
  for (feature in 1:n) {
    for (split in unique(X[, feature])) {
      left_indices <- X[, feature] <= split
      right_indices <- X[, feature] > split
      left <- y[left_indices]
      right <- y[right_indices]
      
      if (length(left) == 0 || length(right) == 0) {
        next
      }
      
      gini <- (length(left) / length(y)) * (1 - sum(left) / length(left)) + 
        (length(right) / length(y)) * (1 - sum(right) / length(right))
      
      if (!is.na(gini) && gini < best_gini) {
        best_gini <- gini
        best_split <- list(feature = feature, value = split, left_indices = left_indices, right_indices = right_indices)
      }
    }
  }
  
  if (is.null(best_split)) {
    return(as.integer(names(sort(table(y), decreasing = TRUE)[1])))
  }
  
  left_tree <- train_tree_classification(X[best_split$left_indices, , drop = FALSE], y[best_split$left_indices], depth - 1)
  right_tree <- train_tree_classification(X[best_split$right_indices, , drop = FALSE], y[best_split$right_indices], depth - 1)
  
  return(list(split = best_split, left = left_tree, right = right_tree))
}

predict_tree_classification <- function(tree, X, ...) {
  predictions <- numeric(nrow(X))
  
  for (i in 1:nrow(X)) {
    node <- tree
    while (is.list(node)) {
      if (X[i, node$split$feature] <= node$split$value) {
        node <- node$left
      } else {
        node <- node$right
      }
    }
    predictions[i] <- node
  }
  
  return(as.integer(predictions))
}

# --- Sieć neuronowa dla klasyfikacji ---
train_nn_classification <- function(train_X, train_y, hidden_neurons = 50, epochs = 2000, learning_rate = 0.0005, ...) {
  train_X <- as.matrix(train_X)
  train_y <- as.numeric(train_y)
  
  mean_X <- apply(train_X, 2, mean)
  sd_X <- apply(train_X, 2, sd)
  train_X <- scale(train_X, center = mean_X, scale = sd_X)
  
  n_features <- ncol(train_X)
  n_samples <- nrow(train_X)
  
  w_hidden <- matrix(rnorm(n_features * hidden_neurons, mean = 0, sd = 0.01), n_features, hidden_neurons)
  b_hidden <- rep(0, hidden_neurons)
  w_output <- matrix(rnorm(hidden_neurons, mean = 0, sd = 0.01), hidden_neurons, 1)
  b_output <- 0
  
  sigmoid <- function(x) 1 / (1 + exp(-x))
  sigmoid_derivative <- function(x) x * (1 - x)
  
  # Wektor do zapisywania wartości straty
  loss_history <- numeric(epochs)
  
  for (epoch in 1:epochs) {
    hidden_layer <- tanh(train_X %*% w_hidden + matrix(b_hidden, n_samples, hidden_neurons, byrow = TRUE))
    output_layer <- sigmoid(hidden_layer %*% w_output + b_output)
    error <- train_y - output_layer
    loss <- mean(error^2)
    loss_history[epoch] <- loss
    
    # Wypisywanie co 100 epok
    if (epoch %% 100 == 0) {
      cat("Epoka:", epoch, "Strata:", loss, "\n")
    }
    
    d_output <- error * sigmoid_derivative(output_layer)
    d_hidden <- d_output %*% t(w_output) * (1 - hidden_layer^2)
    
    w_output <- w_output + t(hidden_layer) %*% d_output * learning_rate
    b_output <- b_output + sum(d_output) * learning_rate
    w_hidden <- w_hidden + t(train_X) %*% d_hidden * learning_rate
    b_hidden <- b_hidden + colSums(d_hidden) * learning_rate
  }
  
  return(list(
    w_hidden = w_hidden,
    b_hidden = b_hidden,
    w_output = w_output,
    b_output = b_output,
    mean_X = mean_X,
    sd_X = sd_X
  ))
}

predict_nn_classification <- function(model, test_X, ...) {
  test_X <- as.matrix(test_X)
  test_X <- scale(test_X, center = model$mean_X, scale = model$sd_X)
  
  hidden_layer <- tanh(test_X %*% model$w_hidden + matrix(model$b_hidden, nrow(test_X), length(model$b_hidden), byrow = TRUE))
  predictions_prob <- 1 / (1 + exp(-(hidden_layer %*% model$w_output + model$b_output)))
  
  return(as.integer(predictions_prob > 0.5))
}


# ============================================================
# Klasyfikacja wieloklasowa
# ============================================================

#knn
train_knn_multiclass <- function(train_X, train_y) {
  return(list(X = train_X, y = train_y))
}

predict_knn_multiclass <- function(model, test_X, k = 10) {
  train_X <- as.matrix(model$X)
  train_y <- model$y
  test_X <- as.matrix(test_X)
  
  n_test <- nrow(test_X)
  predictions <- numeric(n_test)
  
  for (i in 1:n_test) {
    distances <- sqrt(rowSums((train_X - matrix(test_X[i, ], nrow = nrow(train_X), ncol = ncol(train_X), byrow = TRUE))^2))
    neighbors <- train_y[order(distances)[1:k]]
    
    predictions[i] <- as.integer(names(sort(table(neighbors), decreasing = TRUE)[1]))
  }
  
  return(predictions)
}

#drzewo
train_tree_multiclass <- function(X, y, depth) {
  if (depth == 0 || length(unique(y)) == 1) {
    return(as.integer(names(sort(table(y), decreasing = TRUE)[1])))
  }
  
  best_split <- NULL
  best_gini <- Inf
  n <- ncol(X)
  
  for (feature in 1:n) {
    for (split in unique(X[, feature])) {
      left_indices <- X[, feature] <= split
      right_indices <- X[, feature] > split
      left <- y[left_indices]
      right <- y[right_indices]
      
      if (length(left) == 0 || length(right) == 0) next
      
      gini <- sum((table(left) / length(left))^2) + sum((table(right) / length(right))^2)
      
      if (!is.na(gini) && gini < best_gini) {
        best_gini <- gini
        best_split <- list(feature = feature, value = split, left_indices = left_indices, right_indices = right_indices)
      }
    }
  }
  
  if (is.null(best_split)) {
    return(as.integer(names(sort(table(y), decreasing = TRUE)[1])))
  }
  
  left_tree <- train_tree_multiclass(X[best_split$left_indices, , drop = FALSE], y[best_split$left_indices], depth - 1)
  right_tree <- train_tree_multiclass(X[best_split$right_indices, , drop = FALSE], y[best_split$right_indices], depth - 1)
  
  return(list(split = best_split, left = left_tree, right = right_tree))
}

predict_tree_multiclass <- function(tree, X) {
  predictions <- numeric(nrow(X))
  
  for (i in 1:nrow(X)) {
    node <- tree
    while (is.list(node)) {
      if (X[i, node$split$feature] <= node$split$value) {
        node <- node$left
      } else {
        node <- node$right
      }
    }
    predictions[i] <- node
  }
  
  return(as.integer(predictions))
}

#sieć neuronowa
softmax <- function(x) {
  exp_x <- exp(x - apply(x, 1, max)) 
  return(exp_x / rowSums(exp_x))
}

adam_update <- function(m, v, g, beta1, beta2, t, learning_rate, epsilon = 1e-8) {
  m <- beta1 * m + (1 - beta1) * g
  v <- beta2 * v + (1 - beta2) * (g^2)
  m_hat <- m / (1 - beta1^t)
  v_hat <- v / (1 - beta2^t)
  return(-learning_rate * m_hat / (sqrt(v_hat) + epsilon))
}

train_nn_multiclass <- function(train_X, train_y, hidden_neurons = 100, epochs = 5000, learning_rate = 0.001) {
  train_X <- as.matrix(train_X)
  train_y <- as.integer(as.character(train_y))  
  
  mean_X <- apply(train_X, 2, mean)
  sd_X <- apply(train_X, 2, sd)
  train_X <- scale(train_X, center = mean_X, scale = sd_X)
  
  labels <- sort(unique(train_y))
  y_one_hot <- matrix(0, nrow = length(train_y), ncol = length(labels))
  for (i in 1:length(train_y)) {
    y_one_hot[i, which(labels == train_y[i])] <- 1
  }
  
  n_features <- ncol(train_X)
  n_samples <- nrow(train_X)
  n_classes <- length(labels)
  
  w_hidden <- matrix(rnorm(n_features * hidden_neurons, mean = 0, sd = sqrt(1 / n_features)), n_features, hidden_neurons)
  b_hidden <- rep(0, hidden_neurons)
  w_output <- matrix(rnorm(hidden_neurons * n_classes, mean = 0, sd = sqrt(1 / hidden_neurons)), hidden_neurons, n_classes)
  b_output <- rep(0, n_classes)
  
  dropout_rate <- 0.2
  momentum <- 0.9
  velocity_w_hidden <- matrix(0, n_features, hidden_neurons)
  velocity_w_output <- matrix(0, hidden_neurons, n_classes)
  
  for (epoch in 1:epochs) {
    hidden_layer <- train_X %*% w_hidden + matrix(b_hidden, n_samples, hidden_neurons, byrow = TRUE)
    hidden_layer <- (hidden_layer - mean(hidden_layer)) / (sd(hidden_layer) + 1e-8)
    
    hidden_layer <- ifelse(hidden_layer > 0, hidden_layer, 0.01 * hidden_layer)
    
    dropout_mask <- matrix(rbinom(n_samples * hidden_neurons, 1, 1 - dropout_rate), n_samples, hidden_neurons)
    hidden_layer <- hidden_layer * dropout_mask
    
    output_layer <- hidden_layer %*% w_output + matrix(b_output, n_samples, n_classes, byrow = TRUE)
    output_probs <- softmax(output_layer)
    
    loss <- -sum(y_one_hot * log(output_probs + 1e-8)) / n_samples
    
    error <- y_one_hot - output_probs
    d_output <- error
    d_hidden <- d_output %*% t(w_output) * ifelse(hidden_layer > 0, 1, 0.01)
    
    g_w_output <- t(hidden_layer) %*% d_output
    g_w_hidden <- t(train_X) %*% d_hidden
    
    velocity_w_output <- momentum * velocity_w_output + learning_rate * g_w_output
    velocity_w_hidden <- momentum * velocity_w_hidden + learning_rate * g_w_hidden
    
    w_output <- w_output + velocity_w_output
    w_hidden <- w_hidden + velocity_w_hidden
    
    if (epoch %% 500 == 0) {
      cat("Epoka:", epoch, "Strata:", loss, "\n")
    }
  }
  
  return(list(
    w_hidden = w_hidden, b_hidden = b_hidden,
    w_output = w_output, b_output = b_output,
    mean_X = mean_X, sd_X = sd_X,
    labels = labels
  ))
}

predict_nn_multiclass <- function(model, test_X) {
  test_X <- as.matrix(test_X)
  test_X <- scale(test_X, center = model$mean_X, scale = model$sd_X)
  
  test_X[is.na(test_X)] <- 0  
  
  # warstwy ukryte
  hidden_layer <- test_X %*% model$w_hidden + matrix(model$b_hidden, nrow = nrow(test_X), ncol = length(model$b_hidden), byrow = TRUE)
  hidden_layer <- (hidden_layer - mean(hidden_layer)) / (sd(hidden_layer) + 1e-8)
  hidden_layer <- ifelse(hidden_layer > 0, hidden_layer, 0.01 * hidden_layer) 
  
  #warstw. ukryte
  output_layer <- hidden_layer %*% model$w_output + matrix(rep(model$b_output, nrow(test_X)), nrow = nrow(test_X), byrow = TRUE)
  output_probs <- softmax(output_layer)
  
  # Przypisanie klas
  predicted_indices <- apply(output_probs, 1, which.max)
  predictions <- model$labels[predicted_indices]  # Bezpośrednio `labels`
  
  return(as.integer(predictions))
}

