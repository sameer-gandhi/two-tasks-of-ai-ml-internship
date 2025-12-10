# House Price Prediction using California Housing Dataset
# CodexIntern AI/ML Task 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data["PRICE"] = housing.target

print("Dataset Shape:", data.shape)
print("\nSample Rows:\n", data.head())

# Basic statistics
print("\nStats:\n", data.describe())

# Price distribution
plt.figure(figsize=(8,5))
plt.hist(data["PRICE"], bins=30, edgecolor='black')
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
plt.imshow(data.corr(), cmap='coolwarm')
plt.colorbar()
plt.title("Correlation Heatmap")
plt.show()

# Split features and target
X = data.drop("PRICE", axis=1)
y = data["PRICE"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MSE:", round(mse, 3))
print("R2 Score:", round(r2, 3))

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()

# Example prediction
sample = X.iloc[0]
sample_scaled = scaler.transform([sample])
prediction = model.predict(sample_scaled)[0]

print("\nPredicted Price for Sample House:", round(prediction, 3))
