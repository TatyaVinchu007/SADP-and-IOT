 #Slip 1B, 10B
 #Write a Python program to prepare Scatter Plot for Iris Dataset. Convert Categorical values in numeric format for a dataset.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
# Create a DataFrame with the feature data
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Assign target names (species) directly to the 'species' column
df['species'] = [iris.target_names[i] for i in iris.target]

# Check the shape of the DataFrame
print("\nShape of the dataset:", df.shape)

# Check the unique species in the dataset
print("\nUnique species in the dataset:")
print(df['species'].value_counts())

# Display a few rows from different parts of the dataset to verify the species assignment
print("\nFirst 5 rows (should show 'setosa'):")
print(df.iloc[0:5].to_string(index=False))  # Remove index for cleaner output

print("\nRows 50-54 (should show 'versicolor'):")
print(df.iloc[50:55].to_string(index=False))

print("\nRows 100-104 (should show 'virginica'):")
print(df.iloc[100:105].to_string(index=False))

# Prepare the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=df, palette='viridis')
plt.title('Scatter Plot of Sepal Length vs. Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species', loc='upper right')
plt.show()
