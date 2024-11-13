#Slip 2-22A, 4-13B
# Write a python program to implement simple Linear Regression for predicting house price. First find all null values in a given dataset and remove them.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from the provided path
df = pd.read_csv('/content/house-prices.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Check for null values in the dataset
null_values = df.isnull().sum()
print("\nNull values in each column:")
print(null_values)

# Remove rows with null values
df = df.dropna()

# Check the shape of the cleaned dataset
print("\nShape of the cleaned dataset:", df.shape)

# Split the dataset into features (X) and target (y)
X = df[['SqFt']]  # Features
y = df['Price']   # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the model's performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMean Squared Error:", mse)
print("R-squared:", r2)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')
plt.title('House Price Prediction')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()
