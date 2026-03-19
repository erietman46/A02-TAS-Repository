import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import json

data = {}
for subject in range(1, 7):
    globals()[f"data_P{subject}"] = []
    for condition in range(1, 7):
        data_temp = np.load(f"./data/python_data/ae2224I_measurement_data_subj{subject}_C{condition}.npz")
        data_e = data_temp["e"]
        data_e_array = np.array(data_e)
        e_rms = []
        for run in data_e_array:
            i_th_item = data_e_array[:, run]
            e_rms_i = np.sqrt(np.mean(i_th_item ** 2))
            e_rms.append(e_rms_i)
        data_u = data_temp["u"]
        data_t = data_temp["t"]
        globals() [f"data_temp{condition}"] = [e_rms, data_u, data_t]
        globals()[f"data_P{subject}"].append(globals() [f"data_temp{condition}"])
    data[f"Pilot{subject}"] = globals()[f"data_P{subject}"]

data = np.array(data)


"""
combinations = (("e", "u"), ("e", "u", "de"), ("e", "u", "du"), ("e", "u", "de", "du"))
X = None
model = AgglomerativeClustering(n_clusters = 2, linkage = "ward")
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = "viridis")
plt.title("Agglomerative Clustering")
plt.show()
"""