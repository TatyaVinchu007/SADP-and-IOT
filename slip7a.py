# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset (replace 'Salary_positions.csv' with the actual path of the dataset)
data = pd.read_csv('Salary_positions.csv')

# Preview the data
print(data.head())

# Step 1: Data Preprocessing
# Drop any missing values (if there are any)
data = data.dropna()

# Extract features (Level) and target variable (Salary)
X = data[['Level']]  # Reshape to a 2D array as expected by scikit-learn
y = data['Salary']

# Step 2: Model Training
# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 3: Prediction
# Predict salary for levels 11 and 12
levels_to_predict = np.array([[11], [12]])
predicted_salaries = model.predict(levels_to_predict)

# Display the predictions
for level, salary in zip([11, 12], predicted_salaries):
    print(f"Predicted salary for level {level}: ${salary:.2f}")
