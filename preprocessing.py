"""
MATLAB Dataset Loader

Loads all .mat files from the dataset folder and organizes the data as:

dataset[subject][condition][signal]

Example:
dataset[1][2]["t"]  -> time vector for subject 1, condition 2
dataset[3][4]["u"]  -> input signal for subject 3, condition 4
"""

import os
import scipy.io

# -------------------------------------------------------
# Path to the folder containing the dataset
# -------------------------------------------------------
folder = "DATA FOLDER"

# Main dictionary that will store the dataset
dataset = {}

# -------------------------------------------------------
# Loop through all .mat files in the folder
# -------------------------------------------------------
for file in os.listdir(folder):

    if file.endswith(".mat"):

        # Example filename:
        # ae2224I_measurement_data_subj3_C2.mat

        parts = file.replace(".mat", "").split("_")

        # Extract subject and condition numbers
        subject = int(parts[-2].replace("subj", ""))
        condition = int(parts[-1].replace("C", ""))

        # Load the .mat file
        path = os.path.join(folder, file)
        data = scipy.io.loadmat(path)

        # Create dictionary entries if they do not exist
        if subject not in dataset:
            dataset[subject] = {}

        dataset[subject][condition] = {}

        # -------------------------------------------------------
        # Store all signals in the file (ignore MATLAB metadata)
        # -------------------------------------------------------
        for key, value in data.items():
            if not key.startswith("__"):
                dataset[subject][condition][key] = value


print(dataset[2][3]['Hpe_FC'])

"""
Example usage:

Access time signal:
print(t = dataset[1][1]["t"])

Access input signal:
u = dataset[2][3]["u"]

Access measurement:
x = dataset[4][2]["x"]
"""