#The data set refers to clients of a wholesale distributor. It includes the annual spending in monetary units on diverse product categories. Using data Wholesale  customer dataset compute agglomerative clustering to find out annual spending  clients in the same region.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('/content/Wholesale customers data.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Select relevant features for clustering
features = df[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']]

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Compute Agglomerative Clustering
n_clusters = 5  # You can adjust this based on your needs
agglo_clustering = AgglomerativeClustering(n_clusters=n_clusters)
clusters = agglo_clustering.fit_predict(features_scaled)

# Add cluster labels to the original dataframe
df['Cluster'] = clusters

# Visualize the clusters using a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Fresh', y='Milk', hue='Cluster', data=df, palette='viridis')
plt.title('Agglomerative Clustering of Wholesale Customers')
plt.xlabel('Fresh Spending')
plt.ylabel('Milk Spending')
plt.legend(title='Cluster')
plt.grid()
plt.show()
