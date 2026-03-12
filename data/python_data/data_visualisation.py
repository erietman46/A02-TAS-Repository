import numpy as np

subject = 1
condition = 3

data = np.load(f"ae2224I_measurement_data_subj{subject}_C{condition}.npz")
print(data.files)  # shows all variables stored

Hpe_FC = data['Hpe_FC']
print(Hpe_FC.shape)