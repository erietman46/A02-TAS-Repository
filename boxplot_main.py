# import matplotlib.pyplot as plt
# import numpy as np

# global_parameters = np.load("global_parameters.npy", allow_pickle=True)

# # Extract parameters for each condition
# parameters_c1 = global_parameters[0]
# parameters_c2 = global_parameters[1]
# parameters_c3 = global_parameters[2]
# parameters_c4 = global_parameters[3]
# parameters_c5 = global_parameters[4]
# parameters_c6 = global_parameters[5]

# # Condition names (full set)
# condition_names_all = [
#     "C1",
#     "C4",
#     "C2 ",
#     "C5 ",
#     "C3 ",
#     "C6 ",
# ]

# # Combine parameters into a list of lists for boxplot (full set)
# all_conditions_all = [
#     parameters_c1,  # C1
#     parameters_c4,  # C4
#     parameters_c2,  # C2
#     parameters_c5,  # C5
#     parameters_c3,  # C3
#     parameters_c6,  # C6
# ]

# # Subset for Km and tau_m only (C4, C5, C6)
# condition_names_motion_only = [
#     "C4",
#     "C5",
#     "C6",
# ]
# all_conditions_motion_only = [
#     parameters_c4,  # C4
#     parameters_c5,  # C5
#     parameters_c6,  # C6
# ]

# # Original column order (kept here so indices match the data)
# all_params_original = [
#     "Kp [-]",
#     "TL [s]",
#     "TI [s]",
#     "tau [s]",
#     "omega_nm [rad/s]",
#     "zeta_nm [-]",
#     "Km [-]",
#     "Tsc1 [-]",
#     "Tsc2 [-]",
#     "Tsc3 [-]",
#     "tau_m [s]",
# ]



# # Parameters we actually want to plot (exclude Tsc1/2/3)
# params_to_plot = [
#     "Kp [-]",
#     "TL [s]",
#     "TI [s]",
#     "tau [s]",
#     "omega_nm [rad/s]",
#     "zeta_nm [-]",
#     "Km [-]",
#     "tau_m [s]",
# ]



# params_units = [
#     "[-]",
#     "[s]",
#     "[s]",
#     "[s]",
#     "[rad/s]",
#     "[-]",
#     "[-]",
#     "[-]",
#     "[-]",
#     "[-]",
#     "[s]",
# ]


# # Map parameter name -> original column index
# param_to_col = {p: i for i, p in enumerate(all_params_original)}


# for param in params_to_plot:
#     col_idx = param_to_col[param]

#     # Only show C4/C5/C6 for Km and tau_m
#     if param in ("Km [-]", "tau_m [s]"):
#         conditions = all_conditions_motion_only
#         labels = condition_names_motion_only
#     else:
#         conditions = all_conditions_all
#         labels = condition_names_all

#     plt.figure(figsize=(11, 6.5))
#     data_to_plot = [condition[:, col_idx] for condition in conditions]

#     plt.boxplot(data_to_plot, tick_labels=labels, showfliers=False)

#     plt.xlabel("Condition", fontsize=18)
#     plt.ylabel(param, fontsize=18)

#     plt.xticks(ha="right", fontsize=18)
#     plt.yticks(fontsize=18)

#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.22)

#     plt.savefig(f"BOXPLOTS/boxplot_{param}.png", dpi=200, bbox_inches="tight")
#     # plt.show()

# print("Boxplot generation successful")



# # # Create boxplots for each parameter across conditions
# # for param in params_to_plot:
# #     col_idx = param_to_col[param]

# #     # Only show C4/C5/C6 for Km and tau_m
# #     if param in ("Km", "tau_m"):
# #         conditions = all_conditions_motion_only
# #         labels = condition_names_motion_only
# #     else:
# #         conditions = all_conditions_all
# #         labels = condition_names_all

# #     plt.figure(figsize=(10, 6))
# #     data_to_plot = [condition[:, col_idx] for condition in conditions]
# #     plt.boxplot(data_to_plot, tick_labels=labels, showfliers=False)
# #     #plt.title(f"Boxplot of {param} across conditions")
# #     plt.xlabel("Condition")
# #     plt.ylabel(param)
# #     plt.xticks(rotation=45)
# #     plt.tight_layout()
# #     plt.savefig(f"BOXPLOTS/boxplot_{param}.png", dpi=200, bbox_inches="tight")
# #     #plt.show()

import matplotlib.pyplot as plt
import numpy as np

global_parameters = np.load("global_parameters.npy", allow_pickle=True)

# Extract parameters for each condition
parameters_c1 = global_parameters[0]
parameters_c2 = global_parameters[1]
parameters_c3 = global_parameters[2]
parameters_c4 = global_parameters[3]
parameters_c5 = global_parameters[4]
parameters_c6 = global_parameters[5]

# Condition names (full set, includes no-motion + motion)
condition_names_all = [
    "C1",
    "C4",
    "C2",
    "C5",
    "C3",
    "C6",
]

# Full condition list
all_conditions_all = [
    parameters_c1,  # C1
    parameters_c4,  # C4
    parameters_c2,  # C2
    parameters_c5,  # C5
    parameters_c3,  # C3
    parameters_c6,  # C6
]

# Motion-only subset for Km and tau_m
condition_names_motion_only = [
    "C4",
    "C5",
    "C6",
]

all_conditions_motion_only = [
    parameters_c4,  # C4
    parameters_c5,  # C5
    parameters_c6,  # C6
]

# Parameter names without units
all_params_original = [
    "Kp",
    "TL",
    "TI",
    "tau",
    "omega_nm",
    "zeta_nm",
    "Km",
    "Tsc1",
    "Tsc2",
    "Tsc3",
    "tau_m",
]

# Parameters to plot
params_to_plot = [
    "Kp",
    "TL",
    "TI",
    "tau",
    "omega_nm",
    "zeta_nm",
    "Km",
    "tau_m",
]

# Units
param_units = {
    "Kp": "[-]",
    "TL": "[s]",
    "TI": "[s]",
    "tau": "[s]",
    "omega_nm": "[rad/s]",
    "zeta_nm": "[-]",
    "Km": "[-]",
    "Tsc1": "[-]",
    "Tsc2": "[-]",
    "Tsc3": "[-]",
    "tau_m": "[s]",
}

# Optional prettier display names
param_display_names = {
    "Kp": "Kp",
    "TL": "TL",
    "TI": "TI",
    "tau": "tau",
    "omega_nm": "omega_nm",
    "zeta_nm": "zeta_nm",
    "Km": "Km",
    "Tsc1": "Tsc1",
    "Tsc2": "Tsc2",
    "Tsc3": "Tsc3",
    "tau_m": "tau_m",
}

# Map parameter name -> original column index
param_to_col = {p: i for i, p in enumerate(all_params_original)}

for param in params_to_plot:
    col_idx = param_to_col[param]

    # Use motion-only subset only for these two parameters
    if param in ("Km", "tau_m"):
        conditions = all_conditions_motion_only
        labels = condition_names_motion_only
    else:
        conditions = all_conditions_all
        labels = condition_names_all

    plt.figure(figsize=(11, 6.5))
    data_to_plot = [condition[:, col_idx] for condition in conditions]

    plt.boxplot(data_to_plot, tick_labels=labels, showfliers=False)

    ylabel_text = f"{param_display_names[param]} {param_units[param]}"
    plt.xlabel("Condition", fontsize=18)
    plt.ylabel(ylabel_text, fontsize=18)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    plt.savefig(f"BOXPLOTS/boxplot_{param}.png", dpi=200, bbox_inches="tight")
    plt.close()

print("Boxplot generation successful")