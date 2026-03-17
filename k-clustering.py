import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

for subject in range(1, 7)
    for condition in range(1, 4)
        data = np.load(f"ae2224I_measurement_data_subj{subject}_C{condition}.npz")
        data_e = data["e"]
        data_u = data["u"]
        data_t = data["t"]
        f"X{subject}_C{condition}" = data['Hpe_FC']

combinations = (("e", "u"), ("e", "u", "de"), ("e", "u", "du"), ("e", "u", "de", "du"))
X = None
model = AgglomerativeClustering(n_clusters = 2, linkage = "ward")
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = "viridis")
plt.title("Agglomerative Clustering")
plt.show()