"""
Here is a code to load all MATLAB files

Loads all .mat files from the dataset folder and organizes the data as:

dataset[subject][condition][signal]

Example:
dataset[1][2]["t"] --> time vector for subject 1, condition 2
dataset[3][4]["u"] --> input signal for subject 3, condition 4
"""

import os
import scipy.io

folder = "DATA FOLDER"

dataset = {} # Dictionary to hold the data organized by subject and condition

for file in os.listdir(folder):

    if file.endswith(".mat"):


        parts = file.replace(".mat", "").split("_")

        subject = int(parts[-2].replace("subj", ""))
        condition = int(parts[-1].replace("C", ""))

        path = os.path.join(folder, file)
        data = scipy.io.loadmat(path)

        if subject not in dataset:
            dataset[subject] = {}

        dataset[subject][condition] = {}

        for key, value in data.items():
            if not key.startswith("__"):
                dataset[subject][condition][key] = value

print(dataset[1][1]["e"])
"""
Example usage:

Access time signal:
print(t = dataset[1][1]["t"])

Access input signal:
u = dataset[2][3]["u"]

Access measurement:
x = dataset[4][2]["x"]
"""