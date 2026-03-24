import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

def dydx_rms(y, x):
    dy = np.diff(y)
    dx = np.diff(x)
    dydx = dy / dx
    return float(np.sqrt(np.mean(dydx ** 2)))

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
        de_dt_rms = []
        for i in range(data_e_array.shape[1]):
            i_th_item = data_e_array[:, i]
            e_rms_i = float(np.sqrt(np.mean(i_th_item ** 2)))
            e_rms.append(e_rms_i)
            de_dt = dydx_rms(i_th_item, data_t_array)
            de_dt_rms.append(de_dt)

        data_u = data_temp["u"]
        data_u_array = np.array(data_u)
        u_rms = []
        du_dt_rms = []

        for i in range(data_u_array.shape[1]):
            i_th_item = data_u_array[:, i]
            u_rms_i = float(np.sqrt(np.mean(i_th_item ** 2)))
            u_rms.append(u_rms_i)
            du_dt = dydx_rms(i_th_item, data_t_array)
            du_dt_rms.append(du_dt)
        data[(f"P{subject}_C{condition}")] = {"e": e_rms, "u": u_rms, "de_dt": de_dt_rms, "du_dt": du_dt_rms}

combinations = (("e", "u"), ("e", "u", "de_dt"), ("e", "u", "du_dt"), ("e", "u", "de_dt", "du_dt"))
Gain = []
Velocity = []
Acceleration = []
for condition in range(1, 7):
    for pilot in range(1, 7):
        for combination in combinations:
            globals() [f"Combination_{combination}_P_{pilot}_C{condition}"] = [data[f"P{pilot}_C{condition}"][feature] for feature in combination]
            if condition == 1 or condition == 4:
                Gain.append(globals()[f"Combination_{combination}_P_{pilot}_C{condition}"])
            elif condition == 2 or condition == 5:
                Velocity.append(globals()[f"Combination_{combination}_P_{pilot}_C{condition}"])
            elif condition == 3 or condition == 6:
                Acceleration.append(globals()[f"Combination_{combination}_P_{pilot}_C{condition}"])

Gain_e_u = [globals()[f"Combination_('e', 'u')_P_{pilot}_C{condition}"] for condition in [1, 4] for pilot in range(1, 7)]
Gain_e_u = np.array(Gain_e_u)
Gain_e_u = reshape(Gain_e_u)
Gain_e_u = scale(Gain_e_u)

Gain_e_u_de_dt = [globals()[f"Combination_('e', 'u', 'de_dt')_P_{pilot}_C{condition}"] for condition in [1, 4] for pilot in range(1, 7)]
Gain_e_u_de_dt = np.array(Gain_e_u_de_dt)
Gain_e_u_de_dt = reshape(Gain_e_u_de_dt)
Gain_e_u_de_dt = scale(Gain_e_u_de_dt)

Gain_e_u_du_dt = [globals()[f"Combination_('e', 'u', 'du_dt')_P_{pilot}_C{condition}"] for condition in [1, 4] for pilot in range(1, 7)]
Gain_e_u_du_dt = np.array(Gain_e_u_du_dt)
Gain_e_u_du_dt = reshape(Gain_e_u_du_dt)
Gain_e_u_du_dt = scale(Gain_e_u_du_dt)

Gain_e_u_de_dt_du_dt = [globals()[f"Combination_('e', 'u', 'de_dt', 'du_dt')_P_{pilot}_C{condition}"] for condition in [1, 4] for pilot in range(1, 7)]
Gain_e_u_de_dt_du_dt = np.array(Gain_e_u_de_dt_du_dt)2
Gain_e_u_de_dt_du_dt = reshape(Gain_e_u_de_dt_du_dt)
Gain_e_u_de_dt_du_dt = scale(Gain_e_u_de_dt_du_dt)

Velocity_e_u = [globals()[f"Combination_('e', 'u')_P_{pilot}_C{condition}"] for condition in [2, 5] for pilot in range(1, 7)]
Velocity_e_u = np.array(Velocity_e_u)
Velocity_e_u = reshape(Velocity_e_u)
Velocity_e_u = scale(Velocity_e_u)


Velocity_e_u_de_dt = [globals()[f"Combination_('e', 'u', 'de_dt')_P_{pilot}_C{condition}"] for condition in [2, 5] for pilot in range(1, 7)]
Velocity_e_u_de_dt = np.array(Velocity_e_u_de_dt)
Velocity_e_u_de_dt = reshape(Velocity_e_u_de_dt)
Velocity_e_u_de_dt = scale(Velocity_e_u_de_dt)

Velocity_e_u_du_dt = [globals()[f"Combination_('e', 'u', 'du_dt')_P_{pilot}_C{condition}"] for condition in [2, 5] for pilot in range(1, 7)]
Velocity_e_u_du_dt = np.array(Velocity_e_u_du_dt)
Velocity_e_u_du_dt = reshape(Velocity_e_u_du_dt)
Velocity_e_u_du_dt = scale(Velocity_e_u_du_dt)

Velocity_e_u_de_dt_du_dt = [globals()[f"Combination_('e', 'u', 'de_dt', 'du_dt')_P_{pilot}_C{condition}"] for condition in [2, 5] for pilot in range(1, 7)]
Velocity_e_u_de_dt_du_dt = np.array(Velocity_e_u_de_dt_du_dt)
Velocity_e_u_de_dt_du_dt = reshape(Velocity_e_u_de_dt_du_dt)
Velocity_e_u_de_dt_du_dt = scale(Velocity_e_u_de_dt_du_dt)

Acceleration_e_u = [globals()[f"Combination_('e', 'u')_P_{pilot}_C{condition}"] for condition in [3, 6] for pilot in range(1, 7)]
Acceleration_e_u = np.array(Acceleration_e_u)
Acceleration_e_u = reshape(Acceleration_e_u)
Acceleration_e_u = scale(Acceleration_e_u)

Acceleration_e_u_de_dt = [globals()[f"Combination_('e', 'u', 'de_dt')_P_{pilot}_C{condition}"] for condition in [3, 6] for pilot in range(1, 7)]
Acceleration_e_u_de_dt = np.array(Acceleration_e_u_de_dt)
Acceleration_e_u_de_dt = reshape(Acceleration_e_u_de_dt)
Acceleration_e_u_de_dt = scale(Acceleration_e_u_de_dt)

Acceleration_e_u_du_dt = [globals()[f"Combination_('e', 'u', 'du_dt')_P_{pilot}_C{condition}"] for condition in [3, 6] for pilot in range(1, 7)]
Acceleration_e_u_du_dt = np.array(Acceleration_e_u_du_dt)
Acceleration_e_u_du_dt = reshape(Acceleration_e_u_du_dt)
Acceleration_e_u_du_dt = scale(Acceleration_e_u_du_dt)

Acceleration_e_u_de_dt_du_dt = [globals()[f"Combination_('e', 'u', 'de_dt', 'du_dt')_P_{pilot}_C{condition}"] for condition in [3, 6] for pilot in range(1, 7)]
Acceleration_e_u_de_dt_du_dt = np.array(Acceleration_e_u_de_dt_du_dt)
Acceleration_e_u_de_dt_du_dt = reshape(Acceleration_e_u_de_dt_du_dt)
Acceleration_e_u_de_dt_du_dt = scale(Acceleration_e_u_de_dt_du_dt)
