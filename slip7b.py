# Slip 7B, 19B
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample weather forecast dataset (you can replace this with actual data or load from a CSV)
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Step 1: Load the data into a DataFrame
df = pd.DataFrame(data)

# Step 2: Preprocess data (Encode categorical features)
# Convert categorical variables into numeric codes
df_encoded = df.apply(lambda col: col.astype('category').cat.codes)

# Define features (X) and target variable (y)
X = df_encoded[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = df_encoded['Play']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train Naive Bayes classifier
nb_model = CategoricalNB()
nb_model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = nb_model.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Sample prediction
sample_data = [[0, 2, 1, 0]]  # Example: Outlook=Sunny, Temperature=Cool, Humidity=Normal, Wind=Weak
sample_pred = nb_model.predict(sample_data)
print("\nPrediction for sample data (Sunny, Cool, Normal, Weak):", 'Play' if sample_pred[0] == 1 else 'No Play')
