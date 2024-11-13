# Slip 4B, 22A
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data: Size of house in square feet vs. Price in thousands of dollars
# Note: Replace this data with real housing data for a realistic model
house_size = np.array([1500, 1700, 1800, 2000, 2100, 2300, 2400, 2500, 2800, 3000])
house_price = np.array([300, 340, 360, 400, 420, 450, 470, 490, 540, 580])

# Reshape the data
house_size = house_size.reshape(-1, 1)
house_price = house_price.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(house_size, house_price, test_size=0.2, random_state=0)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Print model coefficients
print("Intercept:", model.intercept_)
print("Slope:", model.coef_)

# Calculate mean squared error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Plot the training data and the regression line
plt.scatter(X_train, y_train, color="blue", label="Training data")
plt.scatter(X_test, y_test, color="green", label="Testing data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression line")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price (thousands of dollars)")
plt.title("Simple Linear Regression - House Price Prediction")
plt.legend()
plt.show()
