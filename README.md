# Machine Learning Algorithms Comparison

This repository contains the implementation of various machine learning algorithms (KNN, Decision Tree, Neural Network) built from scratch in R. The project aims to compare these models on different problem types, using cross-validation and hyperparameter tuning to evaluate their performance and stability. The models were compared to popular R machine learning packages.

## Project Overview

This project includes custom implementations of three key machine learning algorithms:
- **K Nearest Neighbours**
- **Decision Tree**
- **Neural Network**

The models were tested on various problems, and their performances were evaluated using cross-validation and hyperparameter optimization techniques.

## Features

- **Custom Implementations**: All models (knn, decision tree, and neural network) are implemented from scratch in R.
- **Cross-Validation**: Cross-validation is used to evaluate the models' performance.
- **Hyperparameter Tuning**: The project explores how different hyperparameters affect model stability and performance.
- **Model Comparison**: We compare the three algorithms based on multiple metrics, such as accuracy, error rates, and stability.

## Getting Started

### Prerequisites

Make sure you have **R** installed. You can download it from [R Project](https://www.r-project.org/).

You may also need some R packages for model evaluation and visualization:

```R
install.packages(readxl)
install.packages(ggplot2)
install.packages(rpart)
install.packages(caret)
install.packages(nnet)
install.packages(gridExtra)
install.packages(reshape2)
