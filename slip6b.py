# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'employee_data.csv' with your actual dataset file)
data = pd.read_csv('employee_data.csv')

# Preview the data
print(data.head())

# Step 1: Data Preprocessing
# Drop rows with missing values
data = data.dropna()

# Select relevant features for clustering (e.g., 'age', 'experience', 'income')
# Modify this based on the actual column names in your dataset
features = data[['age', 'experience', 'income']]

# Step 2: Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3: Apply K-means Clustering
# Define the number of clusters (e.g., 3 for low, medium, and high-income groups)
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['income_cluster'] = kmeans.fit_predict(scaled_features)

# Step 4: Evaluate Clustering with the Elbow Method (optional)
# This helps determine the optimal number of clusters
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.show()

# Step 5: Visualize Clusters
# Create a pairplot to visualize the clustering
sns.pairplot(data, hue='income_cluster', vars=['age', 'experience', 'income'], palette='viridis')
plt.show()

# Step 6: Summary of the clusters
# Get the mean values of each cluster to interpret income groups
print(data.groupby('income_cluster')[['age', 'experience', 'income']].mean())
