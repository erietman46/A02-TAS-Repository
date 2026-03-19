import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

data = {}
for subject in range(1, 7):
    globals()[f"data_P{subject}"] = []
    for condition in range(1, 7):
        data = np.load(f"./data/python_data/ae2224I_measurement_data_subj{subject}_C{condition}.npz")
        data_e = data["e"]
        data_u = data["u"]
        data_t = data["t"]
        globals() [f"data_temp{condition}"] = [data_e, data_u, data_t]
        globals()[f"data_P{subject}"].append(globals() [f"data_temp{condition}"])
    data[f"Pilot{subject}"] = globals()[f"data_P{subject}"]
print(data)


"""
combinations = (("e", "u"), ("e", "u", "de"), ("e", "u", "du"), ("e", "u", "de", "du"))
X = None
model = AgglomerativeClustering(n_clusters = 2, linkage = "ward")
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = "viridis")
plt.title("Agglomerative Clustering")
plt.show()
"""