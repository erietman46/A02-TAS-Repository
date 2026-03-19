import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

def dydx_rms(y, x):
    dy = np.diff(y)
    dx = np.diff(x)
    dydx =dy / dx
    return float(np.sqrt(np.mean(dydx ** 2)))

data = {}
for subject in range(1, 7):
    print(f"Processing subject {subject}...")

    for condition in range(1, 7):
        print(f"  Processing condition {condition}...")
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
        print(f"u: {u_rms}, du/dt: {du_dt_rms}, e: {e_rms}, de/dt: {de_dt_rms}")


data = np.array(data)

combinations = (("e", "u"), ("e", "u", "de_dt"), ("e", "u", "du_dt"), ("e", "u", "de_dt", "du_dt"))