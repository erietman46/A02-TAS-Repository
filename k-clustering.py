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
        data_temp = np.load(f"./data/python_data/ae2224I_measurement_data_subj{subject}_C{condition}.npz")
        data_t = data_temp["t"]
        data_t_array = np.array(data_t)
        data_e = data_temp["e"]
        data_e_array = np.array(data_e)
        e_rms = []

        for i in range(data_e_array.shape[1]):
            i_th_item = data_e_array[:, i]
            e_rms_i = np.sqrt(np.mean(i_th_item ** 2))
            e_rms.append(e_rms_i)
            de = np.diff(i_th_item)
            dt = np.diff(data_t_array)
            de_dt = de / dt
            de_dt = np.sqrt(np.mean(de_dt ** 2))

        data_u = data_temp["u"]
        data_u_array = np.array(data_u)
        u_rms = []

        for i in range(data_u_array.shape[1]):
            i_th_item = data_u_array[:, i]
            u_rms_i = np.sqrt(np.mean(i_th_item ** 2))
            u_rms.append(u_rms_i)
            du = np.diff(i_th_item)
            dt = np.diff(data_t_array)
            du_dt = du / dt
            du_dt = np.sqrt(np.mean(du_dt ** 2))

        globals() [f"data_temp{condition}"] = [e_rms, u_rms, de_dt, du_dt]
        globals()[f"data_P{subject}"].append(globals() [f"data_temp{condition}"])

    data[f"Pilot{subject}"] = globals()[f"data_P{subject}"]

data = np.array(data)


"""
combinations = (("e", "u"), ("e", "u", "de_dt"), ("e", "u", "du_dt"), ("e", "u", "de_dt", "du_dt"))
X = None
model = AgglomerativeClustering(n_clusters = 2, linkage = "ward")
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = "viridis")
plt.title("Agglomerative Clustering")
plt.show()
"""