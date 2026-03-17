import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

combinations = (("e", "u"), ("e", "u", "de"), ("e", "u", "du"), ("e", "u", "de", "du"))
X = None
model = AgglomerativeClustering(n_clusters = 2, linkage = "ward")
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = "viridis")
plt.title("Agglomerative Clustering")
plt.show()