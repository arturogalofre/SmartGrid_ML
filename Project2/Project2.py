# 5,6,7,8
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import numpy as np

# Load the Diabetes dataset
diabetes = load_diabetes()

# Split the data into train, validation and test sets
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=0)

# Define the Lasso model
lasso = Lasso(alpha=0.1, max_iter=10000)

# Train the model on the training set
lasso.fit(X_train, y_train)

# Get the coefficients of the trained model
coefs = lasso.coef_

# Get the indices of the non-zero coefficients
non_zero_coefs = np.where(coefs != 0)[0]

# Get the indices of the top 5 features with the largest coefficients
largest_coefs = np.argsort(np.abs(coefs))[-5:]

# Create a binary mask to select the top 5 features
mask = np.zeros_like(coefs, dtype=bool)
mask[largest_coefs] = True

# Apply the mask to the coefficients
coefs = coefs[mask]

# Get the list of features with non-zero coefficients
feature_names = diabetes.feature_names
selected_features = list(np.array(feature_names)[np.nonzero(mask)])


print("Selected features:", selected_features)
print("Coefficients:", coefs)

# Apply the same feature selection to the test set
score = lasso.score(X_test, y_test)
print("Score on test set:", score)

################


# Load the diabetes dataset
diabetes = load_diabetes()

# Split the data into training, validation, and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, random_state=42)

# Define the hyperparameters and their new values
hyperparams = {'alpha': 0.01,
               'fit_intercept': True,
               'normalize': True,
               'max_iter': 1000}

# Create a new Ridge model with the specified hyperparameters
ridge = Ridge(alpha=hyperparams['alpha'],
              fit_intercept=hyperparams['fit_intercept'],
              normalize=hyperparams['normalize'],
              max_iter=hyperparams['max_iter'])

# Fit the Ridge model on the training and validation data
ridge.fit(X_trainval, y_trainval)

# Select the top 5 features based on the absolute value of their coefficients
features = np.argsort(np.abs(ridge.coef_))[::-1][:5]

# Print the selected features and their coefficients
print("Selected features and coefficients:")
for feature, coef in zip(features, ridge.coef_[features]):
    print(diabetes.feature_names[feature], coef)

# Evaluate the model on the test data
y_pred_test = ridge.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
print("Test MSE:", mse_test)

###################################

# Load the diabetes dataset
diabetes = load_diabetes()

# Split the data into training and test sets (70/30 split) with random_state=0
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.3, random_state=0)

# Normalize the data to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the Linear Lasso model with L1 regularization and alpha=0.1
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

# Get the coefficients and sort them in descending order
coef = lasso.coef_
coef_abs = abs(coef)
sorted_coef_idx = coef_abs.argsort()[::-1]

# Get the top 5 features with the highest coefficients
top_features = [diabetes.feature_names[i] for i in sorted_coef_idx[:5]]
top_coefficients = coef[sorted_coef_idx][:5]

# Print the top 5 features and their coefficients
print("Top 5 features selected by Linear Lasso with L1 regularization:")
for feature, coef in zip(top_features, top_coefficients):
    print(feature, ": ", coef)


######### 9, 10, 11, 12


# Load the diabetes dataset
diabetes = load_diabetes()

# Split the data into training, validation, and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

# Create a decision tree regressor model
model = DecisionTreeRegressor(random_state=42)

# Perform a grid search to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5,
                           scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_trainval, y_trainval)

# Print the best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

# Use the best hyperparameters to fit the model on the training and validation data
best_model = DecisionTreeRegressor(**grid_search.best_params_, random_state=42)
best_model.fit(X_trainval, y_trainval)

# Select the top 5 features based on the feature importance scores
feature_importances = best_model.feature_importances_
features = np.argsort(feature_importances)[::-1][:5]

# Print the selected features and their importance scores
print("Selected features and importance scores:")
for feature, importance in zip(features, feature_importances[features]):
    print(diabetes.feature_names[feature], importance)

# Evaluate the model on the test data
y_pred_test = best_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
print("Test MSE:", mse_test)


##################


# Load the diabetes dataset
diabetes = load_diabetes()

# Split the data into training, validation, and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, random_state=42)

# Define the ElasticNet regression model
enet = ElasticNet()

# Define the hyperparameter grid to search over
param_grid = {'alpha': [0.1, 1, 10, 100, 1000],
              'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
              'fit_intercept': [True, False],
              'normalize': [True, False],
              'max_iter': [500, 1000, 5000, 10000]}


# Perform grid search cross-validation on the training and validation data
grid = GridSearchCV(enet, param_grid=param_grid, cv=5,
                    scoring='neg_mean_squared_error')
grid.fit(X_trainval, y_trainval)

# Get the best ElasticNet model based on the hyperparameters chosen
enet = grid.best_estimator_

# Select the top 5 features based on their coefficient magnitudes
feature_coeffs = list(zip(diabetes.feature_names, enet.coef_))
feature_coeffs.sort(key=lambda x: abs(x[1]), reverse=True)
selected_features = [f[0] for f in feature_coeffs[:5]]

# Print the selected features and their coefficients
print("Selected features and coefficients:")
for feature, coef in feature_coeffs:
    if feature in selected_features:
        print(f"{feature}: {coef}")

# Evaluate the model on the test data and print the mean squared error
y_pred_test = enet.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
print("Test MSE:", mse_test)


####################


# Load the breast cancer dataset
cancer = load_breast_cancer()

# Shuffle and split the dataset into training, validation, and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Use logistic regression with L1 regularization to select a maximum of 5 good features
l1_logreg = LogisticRegression(penalty='l1', solver='liblinear')
l1_logreg.fit(X_train, y_train)

# Select the top 5 features based on their coefficient magnitudes
selector = SelectFromModel(l1_logreg, max_features=5)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)

# Train and evaluate a logistic regression classifier using the selected features
logreg = LogisticRegression()
logreg.fit(X_train_selected, y_train)
y_pred_val = logreg.predict(X_val_selected)
val_accuracy = accuracy_score(y_val, y_pred_val)
print("Validation accuracy with selected features:", val_accuracy)

# Determine a maximum of 5 features that do the best job of classifying examples as benign or malignant using logistic regression
best_features = []
for i in range(5):
    best_accuracy = 0
    for feature in cancer.feature_names:
        if feature not in best_features:
            X_train_i = X_train[:, cancer.feature_names.tolist().index(
                feature)].reshape(-1, 1)
            X_val_i = X_val[:, cancer.feature_names.tolist().index(
                feature)].reshape(-1, 1)
            logreg_i = LogisticRegression()
            logreg_i.fit(X_train_i, y_train)
            y_pred_val_i = logreg_i.predict(X_val_i)
            accuracy_i = accuracy_score(y_val, y_pred_val_i)
            if accuracy_i > best_accuracy:
                best_feature = feature
                best_accuracy = accuracy_i
    best_features.append(best_feature)
    print(f"{i+1}. Best feature: {best_feature}, Validation accuracy: {best_accuracy}")


#################


# Load the breast cancer dataset
cancer = load_breast_cancer()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    test_size=0.2,
                                                    random_state=42)

# Define the model
model = DecisionTreeRegressor()

# Define the grid of hyperparameters to search over
param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8],
    'min_samples_leaf': [1, 2, 3, 4, 5],
}

# Define the grid search object
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

# Use the best hyperparameters to fit the model
model = DecisionTreeRegressor(**grid_search.best_params_)
model.fit(X_train, y_train)

# Get the predictions on the test data
y_pred = model.predict(X_test)

# Print the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)

# Select the top 5 most important features
importances = model.feature_importances_
top_indices = np.argsort(importances)[-5:]
top_features = cancer.feature_names[top_indices]

print("Top 5 features:")
for feature in top_features:
    print(feature)

# Print the score of the final model
score = model.score(X_test, y_test)
print("Score: ", score)

# Get the residuals on the test data
residuals = y_test - y_pred

# Get the index of the worst prediction
worst_prediction_index = np.argmax(np.abs(residuals))

# Print the features and target value for the instance with the worst prediction
print("Worst prediction:")
print("Features: ", X_test[worst_prediction_index])
print("Target value: ", y_test[worst_prediction_index])
print("Predicted value: ", y_pred[worst_prediction_index])

# Get the index of the best prediction
best_prediction_index = np.argmin(np.abs(residuals))

# Print the features and target value for the instance with the best prediction
print("Best prediction:")
print("Features: ", X_test[best_prediction_index])
print("Target value: ", y_test[best_prediction_index])
print("Predicted value: ", y_pred[best_prediction_index])
