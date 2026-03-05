import os
from scipy.io import loadmat
import numpy as np

mat_folder = os.path.join("matlab_data")
output_folder = os.path.join("python_data")

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

metadata_keys = {"__header__", "__version__", "__globals__"}

for i in range(1, 6):
    for j in range(1,6):
        mat_filename = f"ae2224I_measurement_data_subj{i}_C{j}.mat"
        mat_path = os.path.join(mat_folder, mat_filename)  
    
        # Load the MATLAB file
        mat_data = loadmat(mat_path)

        # Remove metadata keys
        mat_data_clean = {k: v for k, v in mat_data.items() if k not in metadata_keys}
    
        # Save processed Python data
        py_filename = f"ae2224I_measurement_data_subj{i}_C{j}.npz"  
        py_path = os.path.join(output_folder, py_filename)
        np.savez(py_path, **mat_data_clean)
    
        print(f"Loaded {mat_filename} and saved to {py_filename}")
