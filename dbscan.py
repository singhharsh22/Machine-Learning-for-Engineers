import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans

# Generate Gaussian noise dataset
np.random.seed(0)
X1 = np.random.normal(loc=[2, 2], scale=0.5, size=(100, 2))  # Cluster 1
X2 = np.random.normal(loc=[6, 6], scale=0.5, size=(100, 2))  # Cluster 2
X3 = np.random.normal(loc=[2, 6], scale=0.5, size=(100, 2))  # Cluster 3
X = np.vstack((X3))  # Combine clusters

# Plot original data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='gray', s=50, edgecolor='k', label="Original Data")
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.2, min_samples=2)
dbscan_labels = dbscan.fit_predict(X)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Plot DBSCAN Results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', s=50, edgecolor='k')
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Cluster Label")

# Plot K-Means Results with Centroids
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50, edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroids")
plt.title("K-Means Clustering with Centroids")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Cluster Label")
plt.legend()

plt.tight_layout()
plt.show()
