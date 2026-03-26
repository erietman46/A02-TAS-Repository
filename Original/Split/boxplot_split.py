import matplotlib.pyplot as plt
import numpy as np

global_parameters = np.load("global_parameters.npy", allow_pickle=True)

#Extract parameters for each condition
parameters_c1 = global_parameters[0]
parameters_c2 = global_parameters[1]
parameters_c3 = global_parameters[2]
parameters_c4 = global_parameters[3]
parameters_c5 = global_parameters[4]
parameters_c6 = global_parameters[5]



#Subjects and conditions
subjects = range(1,7)
conditions = range(1,7)
condition_names = [
    "C1 (P, no motion)",
    "C2 (V, no motion)",
    "C3 (A, no motion)",
    "C4 (P, motion)",
    "C5 (V, motion)",
    "C6 (A, motion)",
]


#Combine parameters into a list of lists for boxplot
all_conditions = [
    parameters_c1,
    parameters_c2,
    parameters_c3,
    parameters_c4,
    parameters_c5,
    parameters_c6,
]

#List of all parameters to plot
all_params = [
    "Kp",
    "TL",
    "TI",
    "tau",
    "omega_nm_vis",
    "zeta_nm_vis",
    "Km",
    "Tsc1",
    "Tsc2",
    "Tsc3",
    "tau_m",
    "omega_nm_vest",
    "zeta_nm_vest",
]


#Create boxplots for each parameter across conditions
for idx, param in enumerate(all_params):
    plt.figure(figsize=(10, 6))
    data_to_plot = [condition[:, idx] for condition in all_conditions]
    plt.boxplot(data_to_plot, tick_labels=condition_names)
    plt.title(f"Boxplot of {param} across conditions")
    plt.xlabel("Condition")
    plt.ylabel(param)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

