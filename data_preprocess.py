import numpy as np
from pathlib import Path

path = Path("./data/python_data")
data = [[None for i in range(7)] for j in range(7)]


for subject in range(1, 7):
    for condition in range(1, 7):
        data[subject-1][condition-1] = np.load(path / f"ae2224I_measurement_data_subj{subject}_C{condition}.npz")
