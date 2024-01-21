import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score


# ---------------------------------- 2 ---------------------------------------------------------
# Money spoent on advertising
mny_spent = 100

# Load advertising data from csv file
df = pd.read_csv("Advertising.csv")

# Split data into features (x) and target (y)
# Reshape array (rows, columns), -1 tells the number of rows is the number of elements in the array
x = df["radio"].values.reshape(-1, 1)
# .values converts dataframe to numpy array as skicit only takes data in numpy array form
y = df["sales"].values

# Create a Linear Regression object
model = LinearRegression()

# Train the model using the training data
model.fit(x, y)
# The fit method is used to train a machine learning model on a training dataset. The fit method takes two required arguments: the first argument is the training data (also called the "features") and the second argument is the target data (also called the "labels").

# Predict the number of unit sales
radio_spend = [[mny_spent]]  # Example amount spent on radio advertising
radio_spend = np.array(radio_spend)
# predicts an output based on the feature mny_spent
sales_prediction = model.predict(radio_spend)

# ---------------------------------- 3 ---------------------------------------------------------

intercept = model.intercept_
coefficient = model.coef_

print("Intercept: ", intercept)
print("Coefficient: ", coefficient)

radio_spend = 23000000
sales_prediction = model.predict([[radio_spend]])
print("Estimated number of units sold: ", sales_prediction)

# ---------------------------------- 4 ---------------------------------------------------------

# The coefficients obtained from the scikit-learn implementation of linear regression should match the results from a gradient descent implementation. Both methods are solving for the same linear regression problem, so the coefficients should be the same.

# As for which method is faster for training the model, scikit-learn uses highly optimized linear algebra libraries such as BLAS, LAPACK, and others, so it is generally faster than a custom gradient descent implementation. However, the speed of training will depend on the size of the dataset and the computational resources available.

# ---------------------------------- 5 ---------------------------------------------------------

# Load the breast cancer dataset
cancer = load_breast_cancer()

# Set the seed value
seed = 123456
rng = np.random.default_rng(seed)

# Select 4 features randomly
idx_feat = (np.floor(30 * rng.uniform(size=4))).astype(int)
X = cancer["data"][:, idx_feat]

# Train the logistic regression model using cross validation
model = LogisticRegression(solver='lbfgs', max_iter=1000)
scores = cross_val_score(model, X, cancer["target"], cv=5)

# Get the feature names for the selected features
feature_names = [cancer["feature_names"][i] for i in idx_feat]

# Print the seed value, feature names, and logistic regression score
print("Seed Value:", seed)
print("Features:", feature_names)
print("Logistic Regression Score:", np.mean(scores))
