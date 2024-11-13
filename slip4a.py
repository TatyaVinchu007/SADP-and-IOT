#Write a python program to implement k-means algorithm on a mall_customers dataset.

from sklearn.cluster import KMeans
import pandas as pd

# Load the Mall Customers dataset
df = pd.read_csv('/content/Mall_Customers.csv')

# Select relevant features for clustering (e.g., Annual Income and Spending Score)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Choose the number of clusters (k)
k = 5  # You can experiment with different values of k

# Initialize and fit the KMeans model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Get cluster labels for each data point
df['Cluster'] = kmeans.labels_

# Visualize the clusters (optional)
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-Means Clustering of Mall Customers')
plt.legend()
plt.show()