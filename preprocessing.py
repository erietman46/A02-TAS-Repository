'''
This file will preprocess the data to be used in the time-domain analysis.
Primarily, it will:
1. Extract the needed arrays from the data
2. Calculate the mean over the various repetitions
3. Present the various pilots and conditions in a clear format for further analysis
'''
import numpy as np

'''
C1 = positions, no motion
C2 = velocity, no motion
C3 = acceleration, no motion
C4 = positions, motion
C5 = velocity, motion
C6 = acceleration, motion
'''

d11 = np.load("data/python_data/ae2224I_measurement_data_subj1_C1.npz")
d12 = np.load("data/python_data/ae2224I_measurement_data_subj1_C2.npz")
d13 = np.load("data/python_data/ae2224I_measurement_data_subj1_C3.npz")
d14 = np.load("data/python_data/ae2224I_measurement_data_subj1_C4.npz")
d15 = np.load("data/python_data/ae2224I_measurement_data_subj1_C5.npz")
d16 = np.load("data/python_data/ae2224I_measurement_data_subj1_C6.npz")

d21 = np.load("data/python_data/ae2224I_measurement_data_subj2_C1.npz")
d22 = np.load("data/python_data/ae2224I_measurement_data_subj2_C2.npz")
d23 = np.load("data/python_data/ae2224I_measurement_data_subj2_C3.npz")
d24 = np.load("data/python_data/ae2224I_measurement_data_subj2_C4.npz")
d25 = np.load("data/python_data/ae2224I_measurement_data_subj2_C5.npz")
d26 = np.load("data/python_data/ae2224I_measurement_data_subj2_C6.npz")

d31 = np.load("data/python_data/ae2224I_measurement_data_subj3_C1.npz")
d32 = np.load("data/python_data/ae2224I_measurement_data_subj3_C2.npz")
d33 = np.load("data/python_data/ae2224I_measurement_data_subj3_C3.npz")
d34 = np.load("data/python_data/ae2224I_measurement_data_subj3_C4.npz")
d35 = np.load("data/python_data/ae2224I_measurement_data_subj3_C5.npz")
d36 = np.load("data/python_data/ae2224I_measurement_data_subj3_C6.npz")

d41 = np.load("data/python_data/ae2224I_measurement_data_subj4_C1.npz")
d42 = np.load("data/python_data/ae2224I_measurement_data_subj4_C2.npz")
d43 = np.load("data/python_data/ae2224I_measurement_data_subj4_C3.npz")
d44 = np.load("data/python_data/ae2224I_measurement_data_subj4_C4.npz")
d45 = np.load("data/python_data/ae2224I_measurement_data_subj4_C5.npz")
d46 = np.load("data/python_data/ae2224I_measurement_data_subj4_C6.npz")

d51 = np.load("data/python_data/ae2224I_measurement_data_subj5_C1.npz")
d52 = np.load("data/python_data/ae2224I_measurement_data_subj5_C2.npz")
d53 = np.load("data/python_data/ae2224I_measurement_data_subj5_C3.npz")
d54 = np.load("data/python_data/ae2224I_measurement_data_subj5_C4.npz")
d55 = np.load("data/python_data/ae2224I_measurement_data_subj5_C5.npz")
d56 = np.load("data/python_data/ae2224I_measurement_data_subj5_C6.npz")

pilot1 = {
    "C1": d11,
    "C2": d12,
    "C3": d13,
    "C4": d14,
    "C5": d15,
    "C6": d16
}
pilot2 = {
    "C1": d21,
    "C2": d22,
    "C3": d23,
    "C4": d24,
    "C5": d25,
    "C6": d26
}
pilot3 = {
    "C1": d31,
    "C2": d32,
    "C3": d33,
    "C4": d34,
    "C5": d35,
    "C6": d36
}
pilot4 = {
    "C1": d41,
    "C2": d42,
    "C3": d43,
    "C4": d44,
    "C5": d45,
    "C6": d46
}
pilot5 = {
    "C1": d51,
    "C2": d52,
    "C3": d53,
    "C4": d54,
    "C5": d55,
    "C6": d56
}

pilots = [pilot1, pilot2, pilot3, pilot4, pilot5]