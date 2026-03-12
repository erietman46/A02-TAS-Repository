import numpy as np
from pathlib import Path

path = Path("./data/python_data")
data = [[None for i in range(6)] for j in range(6)]
data = np.array(data)


for subject in range(1, 7):
    for condition in range(1, 7):
        data[subject-1][condition-1] = np.load(path / f"ae2224I_measurement_data_subj{subject}_C{condition}.npz")

for subject in range(1, 7):
    for condition in range(1, 7):
        npz_file = data[subject-1][condition-1]
        data[subject-1][condition-1] = {key: npz_file[key] for key in npz_file.files}

