
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

# # Condition names (full set, includes no-motion + motion)
# condition_names_all = [
#     "C1",
#     "C4",
#     "C2",
#     "C5",
#     "C3",
#     "C6",
# ]

# # Full condition list
# all_conditions_all = [
#     parameters_c1,  # C1
#     parameters_c4,  # C4
#     parameters_c2,  # C2
#     parameters_c5,  # C5
#     parameters_c3,  # C3
#     parameters_c6,  # C6
# ]

# # Motion-only subset for Km and tau_m
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

# # Parameter names without units
# all_params_original = [
#     "Kp",
#     "TL",
#     "TI",
#     "tau",
#     "omega_nm",
#     "zeta_nm",
#     "Km",
#     "Tsc1",
#     "Tsc2",
#     "Tsc3",
#     "tau_m",
# ]

# # Parameters to plot
# params_to_plot = [
#     "Kp",
#     "TL",
#     "TI",
#     "tau",
#     "omega_nm",
#     "zeta_nm",
#     "Km",
#     "tau_m",
# ]

# # Units
# param_units = {
#     "Kp": "[-]",
#     "TL": "[s]",
#     "TI": "[s]",
#     "tau": "[s]",
#     "omega_nm": "[rad/s]",
#     "zeta_nm": "[-]",
#     "Km": "[-]",
#     "Tsc1": "[-]",
#     "Tsc2": "[-]",
#     "Tsc3": "[-]",
#     "tau_m": "[s]",
# }

# # Optional prettier display names
# param_display_names = {
#     "Kp": "Kp",
#     "TL": "TL",
#     "TI": "TI",
#     "tau": "tau",
#     "omega_nm": "omega_nm",
#     "zeta_nm": "zeta_nm",
#     "Km": "Km",
#     "Tsc1": "Tsc1",
#     "Tsc2": "Tsc2",
#     "Tsc3": "Tsc3",
#     "tau_m": "tau_m",
# }

# # Map parameter name -> original column index
# param_to_col = {p: i for i, p in enumerate(all_params_original)}

# for param in params_to_plot:
#     col_idx = param_to_col[param]

#     # Use motion-only subset only for these two parameters
#     if param in ("Km", "tau_m"):
#         conditions = all_conditions_motion_only
#         labels = condition_names_motion_only
#     else:
#         conditions = all_conditions_all
#         labels = condition_names_all

#     plt.figure(figsize=(11, 6.5))
#     data_to_plot = [condition[:, col_idx] for condition in conditions]

#     plt.boxplot(data_to_plot, tick_labels=labels, showfliers=False)

#     ylabel_text = f"{param_display_names[param]} {param_units[param]}"
#     plt.xlabel("Condition", fontsize=18)
#     plt.ylabel(ylabel_text, fontsize=18)

#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)

#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.22)

#     plt.savefig(f"BOXPLOTS/boxplot_{param}.png", dpi=200, bbox_inches="tight")
#     plt.close()

# print("Boxplot generation successful")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =============================================================================
# Settings
# =============================================================================
OUTPUT_DIR = Path("BOXPLOTS")
OUTPUT_DIR.mkdir(exist_ok=True)

SAVE_FIGURES = True
SHOW_FIGURES = False
FIG_DPI = 200

# =============================================================================
# Plot style
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 14,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# =============================================================================
# Load data
# =============================================================================
global_parameters = np.load("global_parameters.npy", allow_pickle=True)

# Extract parameters for each condition
parameters_c1 = global_parameters[0]
parameters_c2 = global_parameters[1]
parameters_c3 = global_parameters[2]
parameters_c4 = global_parameters[3]
parameters_c5 = global_parameters[4]
parameters_c6 = global_parameters[5]

# =============================================================================
# Condition names / data
# =============================================================================
condition_names_all = ["C1", "C4", "C2", "C5", "C3", "C6"]
all_conditions_all = [
    parameters_c1,
    parameters_c4,
    parameters_c2,
    parameters_c5,
    parameters_c3,
    parameters_c6,
]

condition_names_motion_only = ["C4", "C5", "C6"]
all_conditions_motion_only = [
    parameters_c4,
    parameters_c5,
    parameters_c6,
]

# =============================================================================
# Parameter definitions
# =============================================================================
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

param_units = {
    "Kp": "[-]",
    "TL": "[s]",
    "TI": "[s]",
    "tau": "[s]",
    "omega_nm": "[rad s$^{-1}$]",
    "zeta_nm": "[-]",
    "Km": "[-]",
    "Tsc1": "[-]",
    "Tsc2": "[-]",
    "Tsc3": "[-]",
    "tau_m": "[s]",
}

param_display_names = {
    "Kp": "Kp",
    "TL": "TL",
    "TI": "TI",
    "tau": r"$\tau$",
    "omega_nm": r"$\omega_{nm}$",
    "zeta_nm": r"$\zeta_{nm}$",
    "Km": "Km",
    "Tsc1": "Tsc1",
    "Tsc2": "Tsc2",
    "Tsc3": "Tsc3",
    "tau_m": r"$\tau_m$",
}

param_to_col = {p: i for i, p in enumerate(all_params_original)}

# =============================================================================
# Boxplot helpers
# =============================================================================
def _style_boxplot(bp):
    for box in bp["boxes"]:
        box.set(color="k", linewidth=1.0)
    for whisker in bp["whiskers"]:
        whisker.set(color="k", linewidth=0.9)
    for cap in bp["caps"]:
        cap.set(color="k", linewidth=0.9)
    for median in bp["medians"]:
        median.set(color="k", linewidth=1.2)
    for flier in bp["fliers"]:
        flier.set(
            marker="o",
            markerfacecolor="white",
            markeredgecolor="k",
            markersize=4,
            linestyle="none",
        )

def setup_boxplot_axis(ax, ylabel: str, labels):
    ax.grid(True, axis="y", which="major", color="0.75")
    ax.grid(True, axis="y", which="minor", color="0.82", linestyle=(0, (1.2, 4.5)))
    ax.minorticks_on()
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Condition")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_box_aspect(1)

# =============================================================================
# Plot all parameters
# =============================================================================
for param in params_to_plot:
    col_idx = param_to_col[param]

    if param in ("Km", "tau_m"):
        conditions = all_conditions_motion_only
        labels = condition_names_motion_only
    else:
        conditions = all_conditions_all
        labels = condition_names_all

    data_to_plot = [condition[:, col_idx] for condition in conditions]

    fig, ax = plt.subplots(figsize=(5.2, 5.0))

    bp = ax.boxplot(
        data_to_plot,
        widths=0.55,
        patch_artist=False,
        showfliers=True
    )
    _style_boxplot(bp)

    ylabel_text = f"{param_display_names[param]} {param_units[param]}"
    setup_boxplot_axis(ax, ylabel_text, labels)

    fig.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(OUTPUT_DIR / f"boxplot_{param}.png", dpi=FIG_DPI, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

print("Boxplot generation successful")