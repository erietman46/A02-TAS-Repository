import numpy as np
import torch 

subject = 1
condition = 1

data = np.load(f"ae2224I_measurement_data_subj{subject}_C{condition}.npz")
print(data.files)
