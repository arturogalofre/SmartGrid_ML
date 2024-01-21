from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Load Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Shuffle and split data into training/validation (80%) and test (20%) sets
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_trainval = scaler.fit_transform(X_trainval)
X_test = scaler.transform(X_test)

# Define a parameter grid to search over
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,), (10, 10), (50, 50), (100, 100)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

# Use GridSearchCV to perform a grid search over the parameter grid
mlp = MLPRegressor(random_state=42, max_iter=2000)
grid_search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_trainval, y_trainval)

# Print the best parameters and the corresponding mean validation score
print("Best parameters: ", grid_search.best_params_)
print("Best score on validation set: {:.2f}".format(grid_search.best_score_))

# Evaluate the best estimator on the test set
best_mlp = grid_search.best_estimator_
test_score = best_mlp.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))


################################################################################################################


# Load the Wisconsin breast cancer dataset
data = load_breast_cancer()

# Define the parameter grid to search over
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['logistic', 'tanh', 'relu'],
    'alpha': [0.001, 0.0001],
}

# Create a 5-fold cross-validation object
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Create an MLP classifier object
mlp = MLPClassifier(max_iter=1000, random_state=42)

# Create a GridSearchCV object to search over the parameter grid
grid_search = GridSearchCV(mlp, param_grid, cv=cv, n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(data.data, data.target)

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters:", grid_search.best_params_)
