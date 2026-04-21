import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

def dydx(y, x):
    dy = np.diff(y)
    dx = np.diff(x)
    dydx = dy / dx
    return dydx

def rms(array):
    rms_value = float(np.sqrt(np.mean(array ** 2)))
    return rms_value

def reshape(array):
    reshaped_array = array.reshape(-1, array.shape[-1])
    return reshaped_array

def scale(array):
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(array)
    return scaled_array

data = {}
for subject in range(1, 7):
    for condition in range(1, 7):

        data_temp = np.load(f"./data/python_data/ae2224I_measurement_data_subj{subject}_C{condition}.npz")
        data_t = data_temp["t"]
        data_t_array = np.array(data_t)
        data_e = data_temp["e"]
        data_e_array = np.array(data_e)
        e_rms = []
        e_stdev = []
        de_dt_rms = []
        de_dt_stdev = []

        for i in range(data_e_array.shape[1]):
            i_th_item = data_e_array[:, i]
            e_rms_i = float(np.sqrt(np.mean(i_th_item ** 2)))
            e_rms.append(e_rms_i)
            e_stdev_i = float(np.std(i_th_item))
            e_stdev.append(e_stdev_i)
            de_dt = dydx(i_th_item, data_t_array)
            de_dt_rms.append(rms(de_dt))
            de_dt_stdev_i = float(np.std(de_dt))
            de_dt_stdev.append(de_dt_stdev_i)

        data_u = data_temp["u"]
        data_u_array = np.array(data_u)
        u_rms = []
        u_stdev = []
        du_dt_rms = []
        du_dt_stdev = []

        for i in range(data_u_array.shape[1]):
            i_th_item = data_u_array[:, i]
            u_rms_i = float(np.sqrt(np.mean(i_th_item ** 2)))
            u_rms.append(u_rms_i)
            u_stdev_i = float(np.std(i_th_item))
            u_stdev.append(u_stdev_i)
            du_dt = dydx(i_th_item, data_t_array)
            du_dt_rms.append(rms(du_dt))
            du_dt_stdev_i = float(np.std(du_dt))
            du_dt_stdev.append(du_dt_stdev_i)
        data[(f"P{subject}_C{condition}")] = {"e": e_rms, "e_stdev": e_stdev, 
                                              "u": u_rms, "u_stdev": u_stdev, 
                                              "de_dt": de_dt_rms, "de_dt_stdev": de_dt_stdev, 
                                              "du_dt": du_dt_rms, "du_dt_stdev": du_dt_stdev}

combinations = (("e", "u", "e_stdev", "u_stdev"), 
                ("e", "u", "de_dt", "e_stdev", "u_stdev", "de_dt_stdev"), 
                ("e", "u", "du_dt", "e_stdev", "u_stdev", "du_dt_stdev"), 
                ("e", "u", "de_dt", "du_dt", "e_stdev", "u_stdev", "de_dt_stdev", "du_dt_stdev"))

for condition in range(1, 7):
    for pilot in range(1, 7):
        for combination in combinations:
            globals() [f"Combination_{combination}_P_{pilot}_C{condition}"] = [data[f"P{pilot}_C{condition}"][feature] for feature in combination]

datasets_list = []

Gain_e_u = [globals()[f"Combination_('e', 'u', 'e_stdev', 'u_stdev')_P_{pilot}_C{condition}"] for condition in [1, 4] for pilot in range(1, 7)]
Gain_e_u = np.array(Gain_e_u)
Gain_e_u = Gain_e_u.transpose(0, 2, 1)
Gain_e_u = reshape(Gain_e_u)
Gain_e_u = scale(Gain_e_u)
datasets_list.append(Gain_e_u)

Gain_e_u_de_dt = [globals()[f"Combination_('e', 'u', 'de_dt', 'e_stdev', 'u_stdev', 'de_dt_stdev')_P_{pilot}_C{condition}"] for condition in [1, 4] for pilot in range(1, 7)]
Gain_e_u_de_dt = np.array(Gain_e_u_de_dt)
Gain_e_u_de_dt = Gain_e_u_de_dt.transpose(0, 2, 1)
Gain_e_u_de_dt = reshape(Gain_e_u_de_dt)
Gain_e_u_de_dt = scale(Gain_e_u_de_dt)
datasets_list.append(Gain_e_u_de_dt)

Gain_e_u_du_dt = [globals()[f"Combination_('e', 'u', 'du_dt', 'e_stdev', 'u_stdev', 'du_dt_stdev')_P_{pilot}_C{condition}"] for condition in [1, 4] for pilot in range(1, 7)]
Gain_e_u_du_dt = np.array(Gain_e_u_du_dt)
Gain_e_u_du_dt = Gain_e_u_du_dt.transpose(0, 2, 1)
Gain_e_u_du_dt = reshape(Gain_e_u_du_dt)
Gain_e_u_du_dt = scale(Gain_e_u_du_dt)
datasets_list.append(Gain_e_u_du_dt)

Gain_e_u_de_dt_du_dt = [globals()[f"Combination_('e', 'u', 'de_dt', 'du_dt', 'e_stdev', 'u_stdev', 'de_dt_stdev', 'du_dt_stdev')_P_{pilot}_C{condition}"] for condition in [1, 4] for pilot in range(1, 7)]
Gain_e_u_de_dt_du_dt = np.array(Gain_e_u_de_dt_du_dt)
Gain_e_u_de_dt_du_dt = Gain_e_u_de_dt_du_dt.transpose(0, 2, 1)
Gain_e_u_de_dt_du_dt = reshape(Gain_e_u_de_dt_du_dt)
Gain_e_u_de_dt_du_dt = scale(Gain_e_u_de_dt_du_dt)
datasets_list.append(Gain_e_u_de_dt_du_dt)

Velocity_e_u = [globals()[f"Combination_('e', 'u', 'e_stdev', 'u_stdev')_P_{pilot}_C{condition}"] for condition in [2, 5] for pilot in range(1, 7)]
Velocity_e_u = np.array(Velocity_e_u)
Velocity_e_u = Velocity_e_u.transpose(0, 2, 1)
Velocity_e_u = reshape(Velocity_e_u)
Velocity_e_u = scale(Velocity_e_u)
datasets_list.append(Velocity_e_u)

Velocity_e_u_de_dt = [globals()[f"Combination_('e', 'u', 'de_dt', 'e_stdev', 'u_stdev', 'de_dt_stdev')_P_{pilot}_C{condition}"] for condition in [2, 5] for pilot in range(1, 7)]
Velocity_e_u_de_dt = np.array(Velocity_e_u_de_dt)
Velocity_e_u_de_dt = Velocity_e_u_de_dt.transpose(0, 2, 1)
Velocity_e_u_de_dt = reshape(Velocity_e_u_de_dt)
Velocity_e_u_de_dt = scale(Velocity_e_u_de_dt)
datasets_list.append(Velocity_e_u_de_dt)

Velocity_e_u_du_dt = [globals()[f"Combination_('e', 'u', 'du_dt', 'e_stdev', 'u_stdev', 'du_dt_stdev')_P_{pilot}_C{condition}"] for condition in [2, 5] for pilot in range(1, 7)]
Velocity_e_u_du_dt = np.array(Velocity_e_u_du_dt)
Velocity_e_u_du_dt = Velocity_e_u_du_dt.transpose(0, 2, 1)
Velocity_e_u_du_dt = reshape(Velocity_e_u_du_dt)
Velocity_e_u_du_dt = scale(Velocity_e_u_du_dt)
datasets_list.append(Velocity_e_u_du_dt)

Velocity_e_u_de_dt_du_dt = [globals()[f"Combination_('e', 'u', 'de_dt', 'du_dt', 'e_stdev', 'u_stdev', 'de_dt_stdev', 'du_dt_stdev')_P_{pilot}_C{condition}"] for condition in [2, 5] for pilot in range(1, 7)]
Velocity_e_u_de_dt_du_dt = np.array(Velocity_e_u_de_dt_du_dt)
Velocity_e_u_de_dt_du_dt = Velocity_e_u_de_dt_du_dt.transpose(0, 2, 1)
Velocity_e_u_de_dt_du_dt = reshape(Velocity_e_u_de_dt_du_dt)
Velocity_e_u_de_dt_du_dt = scale(Velocity_e_u_de_dt_du_dt)
datasets_list.append(Velocity_e_u_de_dt_du_dt)

Acceleration_e_u = [globals()[f"Combination_('e', 'u', 'e_stdev', 'u_stdev')_P_{pilot}_C{condition}"] for condition in [3, 6] for pilot in range(1, 7)]
Acceleration_e_u = np.array(Acceleration_e_u)
Acceleration_e_u = Acceleration_e_u.transpose(0, 2, 1)
Acceleration_e_u = reshape(Acceleration_e_u)
Acceleration_e_u = scale(Acceleration_e_u)
datasets_list.append(Acceleration_e_u)

Acceleration_e_u_de_dt = [globals()[f"Combination_('e', 'u', 'de_dt', 'e_stdev', 'u_stdev', 'de_dt_stdev')_P_{pilot}_C{condition}"] for condition in [3, 6] for pilot in range(1, 7)]
Acceleration_e_u_de_dt = np.array(Acceleration_e_u_de_dt)
Acceleration_e_u_de_dt = Acceleration_e_u_de_dt.transpose(0, 2, 1)
Acceleration_e_u_de_dt = reshape(Acceleration_e_u_de_dt)
Acceleration_e_u_de_dt = scale(Acceleration_e_u_de_dt)
datasets_list.append(Acceleration_e_u_de_dt)

Acceleration_e_u_du_dt = [globals()[f"Combination_('e', 'u', 'du_dt', 'e_stdev', 'u_stdev', 'du_dt_stdev')_P_{pilot}_C{condition}"] for condition in [3, 6] for pilot in range(1, 7)]
Acceleration_e_u_du_dt = np.array(Acceleration_e_u_du_dt)
Acceleration_e_u_du_dt = Acceleration_e_u_du_dt.transpose(0, 2, 1)
Acceleration_e_u_du_dt = reshape(Acceleration_e_u_du_dt)
Acceleration_e_u_du_dt = scale(Acceleration_e_u_du_dt)
datasets_list.append(Acceleration_e_u_du_dt)

Acceleration_e_u_de_dt_du_dt = [globals()[f"Combination_('e', 'u', 'de_dt', 'du_dt', 'e_stdev', 'u_stdev', 'de_dt_stdev', 'du_dt_stdev')_P_{pilot}_C{condition}"] for condition in [3, 6] for pilot in range(1, 7)]
Acceleration_e_u_de_dt_du_dt = np.array(Acceleration_e_u_de_dt_du_dt)
Acceleration_e_u_de_dt_du_dt = Acceleration_e_u_de_dt_du_dt.transpose(0, 2, 1)
Acceleration_e_u_de_dt_du_dt = reshape(Acceleration_e_u_de_dt_du_dt)
Acceleration_e_u_de_dt_du_dt = scale(Acceleration_e_u_de_dt_du_dt)
datasets_list.append(Acceleration_e_u_de_dt_du_dt)

hierarchical_clustering = AgglomerativeClustering(n_clusters=2, linkage = "ward")
labels_list = []
titles_list = []

labels_Gain_e_u = hierarchical_clustering.fit_predict(Gain_e_u)
labels_list.append(labels_Gain_e_u)
titles_list.append("Gain (e, u)")
labels_Gain_e_u_de_dt = hierarchical_clustering.fit_predict(Gain_e_u_de_dt)
labels_list.append(labels_Gain_e_u_de_dt)
titles_list.append("Gain (e, u, de/dt)")
labels_Gain_e_u_du_dt = hierarchical_clustering.fit_predict(Gain_e_u_du_dt)
labels_list.append(labels_Gain_e_u_du_dt)
titles_list.append("Gain (e, u, du/dt)")
labels_Gain_e_u_de_dt_du_dt = hierarchical_clustering.fit_predict(Gain_e_u_de_dt_du_dt)
labels_list.append(labels_Gain_e_u_de_dt_du_dt)
titles_list.append("Gain (e, u, de/dt, du/dt)")

labels_Velocity_e_u = hierarchical_clustering.fit_predict(Velocity_e_u)
labels_list.append(labels_Velocity_e_u)
titles_list.append("Velocity (e, u)")
labels_Velocity_e_u_de_dt = hierarchical_clustering.fit_predict(Velocity_e_u_de_dt)
labels_list.append(labels_Velocity_e_u_de_dt)
titles_list.append("Velocity (e, u, de/dt)")
labels_Velocity_e_u_du_dt = hierarchical_clustering.fit_predict(Velocity_e_u_du_dt)
labels_list.append(labels_Velocity_e_u_du_dt)
titles_list.append("Velocity (e, u, du/dt)")
labels_Velocity_e_u_de_dt_du_dt = hierarchical_clustering.fit_predict(Velocity_e_u_de_dt_du_dt)
labels_list.append(labels_Velocity_e_u_de_dt_du_dt)
titles_list.append("Velocity (e, u, de/dt, du/dt)")

labels_Acceleration_e_u = hierarchical_clustering.fit_predict(Acceleration_e_u)
labels_list.append(labels_Acceleration_e_u)
titles_list.append("Acceleration (e, u)")
labels_Acceleration_e_u_de_dt = hierarchical_clustering.fit_predict(Acceleration_e_u_de_dt)
labels_list.append(labels_Acceleration_e_u_de_dt)
titles_list.append("Acceleration (e, u, de/dt)")
labels_Acceleration_e_u_du_dt = hierarchical_clustering.fit_predict(Acceleration_e_u_du_dt)
labels_list.append(labels_Acceleration_e_u_du_dt)
titles_list.append("Acceleration (e, u, du/dt)")
labels_Acceleration_e_u_de_dt_du_dt = hierarchical_clustering.fit_predict(Acceleration_e_u_de_dt_du_dt)
labels_list.append(labels_Acceleration_e_u_de_dt_du_dt)
titles_list.append("Acceleration (e, u, de/dt, du/dt)")

fig, axes = plt.subplots(4, 3, figsize=(12, 22))
pca = PCA(n_components=2)
for ax, X, labels, title in zip(axes.flatten(), datasets_list, labels_list, titles_list):
    X_2d_plot = pca.fit_transform(X)
    for cluster, color in zip([0, 1], ['blue', 'red']):
        mask = labels == cluster
        ax.scatter(X_2d_plot[mask, 0], X_2d_plot[mask, 1], c=color, label=f'Cluster {cluster + 1}')
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.4, wspace=0.3)
plt.show()

true_labels = np.repeat([0] * 6 + [1] * 6, 5) 
scores_list = []
for labels, title in zip(labels_list, titles_list):
    score = adjusted_rand_score(true_labels, labels)
    scores_list.append(score)
    print(f"{title}: {score:.3f}")