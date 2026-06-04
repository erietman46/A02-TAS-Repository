# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from error_pdf import error_pdf
# from RMS_error import RMS_error
# from derivative_e_RMS import RMS_DERe
# from contributions import contributions
#
#
# def generate_individual_plots(data_dict, numbers, p_values, effect_sizes, units_dict):
#     """
#     Generates and saves a separate figure for each metric in the data_dict.
#     """
#     sns.set_theme(style="whitegrid")
#     metrics = list(data_dict.keys())
#
#     for metric in metrics:
#         # Create a new figure for each metric
#         plt.figure(figsize=(6, 6))
#
#         no_mo = data_dict[metric][0]
#         mo = data_dict[metric][1]
#         plot_data = [no_mo, mo]
#         labels = ['Fixed-Base', 'Motion-Base']
#
#         # 1. Draw the Boxplot
#         sns.boxplot(data=plot_data, color="white", width=0.5, showfliers=False)
#
#         # 2. Add Strip Plot (Individual dots)
#         sns.stripplot(data=plot_data, color="black", size=7, jitter=False, alpha=0.8)
#
#         # 3. Draw connecting lines between pairs
#         for j in range(len(no_mo)):
#             plt.plot([0, 1], [no_mo[j], mo[j]], color='gray', linestyle='--', linewidth=1, alpha=0.5)
#
#         # 4. Annotations (Significance and Effect Size)
#         metric_label = numbers[metric]
#         p_val = p_values[metric]
#         es = effect_sizes[metric]
#         stats_text = f"{metric_label}\np = {p_val:.4f}\nd = {es:.2f}"
#
#         # Calculate text placement
#         y_max = max(max(no_mo), max(mo))
#         y_range = y_max - min(min(no_mo), min(mo))
#         plt.text(0.5, y_max + (y_range * 0.1), stats_text,
#                  ha='center', va='bottom', fontsize=12, fontweight='bold')
#
#         # Formatting
#         unit = units_dict.get(metric, "")
#         #plt.title(f"Acceleration Task: {metric}", fontsize=14, fontweight='bold')
#         plt.xticks([0, 1], labels)
#         plt.ylabel(f"[{unit}]", fontsize=12)
#
#         # Save each plot with a unique filename
#         # We replace spaces with underscores for the filename
#         filename = f"{metric.replace(' ', '_')}_accel_results.png"
#         plt.tight_layout()
#         plt.savefig(filename, dpi=300)
#         print(f"Saved: {filename}")
#
#         plt.show()  # Shows the window for the current metric before moving to the next
#
#
# # --- EXAMPLE USAGE ---
# # Data for the Acceleration condition only
# my_units = {
#     "RMS Error": "degrees",
#     "RMS Error Deriv": "degrees/s",
#     "1-Sigma Interval": "degrees"
# }
#
# def create_accel_data_dict(rms_err, rms_deriv, sigma_int):
#     """
#     Extracts the Acceleration columns (index 2 and 5)
#     from the (6,6) metric arrays.
#     """
#
#     data_dict = {
#         "RMS Error": [
#             rms_err[:, 2],  # Column 2: Acceleration No-Motion
#             rms_err[:, 5]  # Column 5: Acceleration Motion
#         ],
#         "RMS Error Deriv": [
#             rms_deriv[:, 2],
#             rms_deriv[:, 5]
#         ],
#         "1-Sigma Interval": [
#             sigma_int[:, 2],
#             sigma_int[:, 5]
#         ]
#     }
#
#     return data_dict
#
# _,_,_,_,sigma_interval_array = error_pdf()
# _,_,rms_error_array = RMS_error()
# _,_,rms_deriv_array = RMS_DERe()
#
# accel_data = create_accel_data_dict(
#     rms_error_array,
#     rms_deriv_array,
#     sigma_interval_array
# )
#
# numbers = {"RMS Error": "(a)", "RMS Error Deriv": "(b)", "1-Sigma Interval": "(c)"}
# p_vals = {"RMS Error": 0.0033, "RMS Error Deriv": 0.0105, "1-Sigma Interval": 0.0012}
# e_sizes = {"RMS Error": -2.14, "RMS Error Deriv": -1.63, "1-Sigma Interval": -2.72}
# generate_individual_plots(accel_data, numbers, p_vals, e_sizes, my_units)
#
# def create_pos_data_dict(dist, target, noise):
#     """
#     Extracts the Acceleration columns (index 2 and 5)
#     from the (6,6) metric arrays.
#     """
#
#     data_dict = {
#         "Contribution Disturbance": [
#             dist[:, 0],  # Column 2: Acceleration No-Motion
#             dist[:, 3]  # Column 5: Acceleration Motion
#         ],
#         "Contribution Target": [
#             target[:, 0],
#             target[:, 3]
#         ],
#         "Contribution Noise": [
#             noise[:, 0],
#             noise[:, 3]
#         ]
#     }
#
#     return data_dict
#
# _,_,_,_,_,_,dist_array,target_array,noise_array = contributions()
# pos_data = create_pos_data_dict(dist_array, target_array, noise_array)
#
# my_units2 = {
#     "Contribution Disturbance": "percent",
#     "Contribution Target": "percent",
#     "Contribution Noise": "percent"
# }
#
# numbers2 = {"Contribution Disturbance": "(a)", "Contribution Target": "(b)", "Contribution Noise": "(c)"}
# p_vals2 = {"Contribution Disturbance": 0.0255, "Contribution Target": 0.0478, "Contribution Noise": 0.0376}
# e_sizes2 = {"Contribution Disturbance": 1.28, "Contribution Target": 1.06, "Contribution Noise": -1.15}
# generate_individual_plots(pos_data, numbers2, p_vals2, e_sizes2, my_units2)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from error_pdf import error_pdf
from RMS_error import RMS_error
from derivative_e_RMS import RMS_DERe
from contributions import contributions

# =============================================================================

# =============================================================================
# GLOBAL STYLE CONFIGURATION (MATCHING TEAM GUIDE)
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 14,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

FIG_DPI = 220


def generate_individual_plots(data_dict, numbers, p_values, effect_sizes, units_dict):
    """
    Generates plots with horizontal-only grids, paired connecting lines,
    and simplified (a), (b), (c) labels.
    """
    metrics = list(data_dict.keys())

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(5, 5))

        no_mo = data_dict[metric][0]
        mo = data_dict[metric][1]
        plot_data = [no_mo, mo]
        labels = ['Fixed-Base', 'Motion-Base']

        # 1. Draw the Boxplot
        sns.boxplot(ax=ax, data=plot_data, color="white", width=0.5,
                    showfliers=False,  # Lines connect to points, so fliers aren't needed
                    boxprops=dict(linewidth=0.8, zorder=2),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    medianprops=dict(color="black", linewidth=1.2))

        # 2. Add Strip Plot (To give the lines something to connect to)
        sns.stripplot(ax=ax, data=plot_data, color="black", size=5, jitter=False, alpha=0.8, zorder=3)

        # 3. RESTORED: Connecting lines between pairs
        # Using a light gray and thin width to keep it from looking cluttered
        for j in range(len(no_mo)):
            ax.plot([0, 1], [no_mo[j], mo[j]], color='0.7', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

        # 4. Setup Grid (Horizontal Only)
        ax.minorticks_on()
        ax.grid(True, axis='y', which="major", color="0.80", linestyle='-')
        ax.grid(True, axis='y', which="minor", color="0.85", linestyle=(0, (1.2, 4.5)))
        ax.grid(False, axis='x')  # Keep vertical lines off

        ax.set_box_aspect(1)

        # 5. Stats Annotation
        p_val = p_values[metric]
        es = effect_sizes[metric]
        stats_text = f"p = {p_val:.4f}\nd = {es:.2f}"

        y_max = max(max(no_mo), max(mo))
        y_min = min(min(no_mo), min(mo))
        y_range = y_max - y_min

        ax.text(0.5, y_max + (y_range * 0.08), stats_text,
                ha='center', va='bottom', fontsize=10)

        # 6. Formatting Labels
        unit = units_dict.get(metric, "")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_ylabel(rf"$\mathrm{{[{unit}]}}$", fontsize=13)

        # 7. Simplified Title: Just (a), (b), or (c)
        # Positioned below the axis per the team style guide
        #ax.set_title(f"{numbers[metric]}", y=-0.25, fontweight="bold", fontsize=14)

        # Save and Show
        filename = f"{metric.replace(' ', '_')}_results.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=FIG_DPI, bbox_inches="tight")
        plt.show()


# --- Usage remains the same as your previous logic ---


# --- INPUT LOGIC (UNCHANGED) ---

# my_units = {
#     "RMS Error": "deg",
#     "RMS Error Deriv": "deg/s",
#     "1-Sigma Interval": "deg"
# }
#
#
# def create_accel_data_dict(rms_err, rms_deriv, sigma_int):
#     data_dict = {
#         "RMS Error": [rms_err[:, 2], rms_err[:, 5]],
#         "RMS Error Deriv": [rms_deriv[:, 2], rms_deriv[:, 5]],
#         "1-Sigma Interval": [sigma_int[:, 2], sigma_int[:, 5]]
#     }
#     return data_dict
#
#
# _, _, _, _, sigma_interval_array = error_pdf()
# _, _, rms_error_array = RMS_error()
# _, _, rms_deriv_array = RMS_DERe()
#
# accel_data = create_accel_data_dict(rms_error_array, rms_deriv_array, sigma_interval_array)
#
# numbers = {"RMS Error": "(a)", "RMS Error Deriv": "(b)", "1-Sigma Interval": "(c)"}
# p_vals = {"RMS Error": 0.0033, "RMS Error Deriv": 0.0105, "1-Sigma Interval": 0.0012}
# e_sizes = {"RMS Error": -2.14, "RMS Error Deriv": -1.63, "1-Sigma Interval": -2.72}
# generate_individual_plots(accel_data, numbers, p_vals, e_sizes, my_units)
#
#
# def create_pos_data_dict(dist, target, noise):
#     data_dict = {
#         "Contribution Disturbance": [dist[:, 0], dist[:, 3]],
#         "Contribution Target": [target[:, 0], target[:, 3]],
#         "Contribution Noise": [noise[:, 0], noise[:, 3]]
#     }
#     return data_dict
#
#
# _, _, _, _, _, _, dist_array, target_array, noise_array = contributions()
# pos_data = create_pos_data_dict(dist_array, target_array, noise_array)
#
# my_units2 = {
#     "Contribution Disturbance": "\%",
#     "Contribution Target": "\%",
#     "Contribution Noise": "\%"
# }
#
# numbers2 = {"Contribution Disturbance": "(a)", "Contribution Target": "(b)", "Contribution Noise": "(c)"}
# p_vals2 = {"Contribution Disturbance": 0.0255, "Contribution Target": 0.0478, "Contribution Noise": 0.0376}
# e_sizes2 = {"Contribution Disturbance": 1.28, "Contribution Target": 1.06, "Contribution Noise": -1.15}
# generate_individual_plots(pos_data, numbers2, p_vals2, e_sizes2, my_units2)


# def generate_plots(data_list, p_vals, e_sizes, units_list, names_list):
#     """
#     Inputs are now lists. Index 0 = RMS Error, Index 1 = RMS Deriv, Index 2 = 1-Sigma.
#     """
#     for i in range(len(data_list)):
#         fig, ax = plt.subplots(figsize=(5, 5))
#
#         # Access list items by index
#         no_mo, mo = data_list[i]
#         p_val = p_vals[i]
#         es = e_sizes[i]
#         unit = units_list[i]
#         name = names_list[i]
#
#         # Automatic labeling: 0 -> (a), 1 -> (b), 2 -> (c)
#         #sublabel = f"({chr(97 + i)})"
#
#         plot_data = [no_mo, mo]
#         labels = ['Fixed-Base', 'Motion-Base']
#
#         # 1. Draw the Boxplot
#         sns.boxplot(ax=ax, data=plot_data, color="white", width=0.5,
#                     showfliers=False, zorder=2,
#                     boxprops=dict(linewidth=0.8),
#                     whiskerprops=dict(linewidth=0.8),
#                     capprops=dict(linewidth=0.8),
#                     medianprops=dict(color="black", linewidth=1.2))
#
#         # 2. Add Strip Plot
#         sns.stripplot(ax=ax, data=plot_data, color="black", size=5, jitter=False, alpha=0.8, zorder=3)
#
#         # 3. Connecting lines
#         for j in range(len(no_mo)):
#             ax.plot([0, 1], [no_mo[j], mo[j]], color='0.7', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
#
#         # 4. Setup Grid (Horizontal Only)
#         ax.minorticks_on()
#         ax.grid(True, axis='y', which="major", color="0.80", linestyle='-')
#         ax.grid(True, axis='y', which="minor", color="0.85", linestyle=(0, (1.2, 4.5)))
#         ax.grid(False, axis='x')
#         ax.set_box_aspect(1)
#
#         # 5. Stats Annotation
#         stats_text = f"p = {p_val:.4f}\nd = {es:.2f}"
#         y_max = max(max(no_mo), max(mo))
#         y_range = y_max - min(min(no_mo), min(mo))
#         ax.text(0.5, y_max + (y_range * 0.08), stats_text, ha='center', va='bottom', fontsize=10)
#
#         # 6. Formatting
#         ax.set_xticks([0, 1])
#         ax.set_xticklabels(labels)
#         ax.set_ylabel(rf"$\mathrm{{[{unit}]}}$", fontsize=13)
#
#         # Use the sublabel (a, b, c) as the title below the plot
#         #ax.set_title(sublabel, y=-0.25, fontweight="bold", fontsize=14)
#
#         # 7. Save and Show
#         filename = f"{name.replace(' ', '_')}_results.png"
#         plt.tight_layout()
#         plt.savefig(filename, dpi=FIG_DPI, bbox_inches="tight")
#         plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(data_list, units_list, names_list):
    """
    data_list: list of 6 arrays, one per condition [C1, C2, C3, C4, C5, C6]
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    unit = units_list
    name = names_list

    # Reorder to C1 C4 C2 C5 C3 C6
    reordered = [data_list[0], data_list[3],
                 data_list[1], data_list[4],
                 data_list[2], data_list[5]]
    labels = ['C1', 'C4', 'C2', 'C5', 'C3', 'C6']

    # 1. Draw the Boxplot with hollow outliers
    # We set showfliers=True and use flierprops to format them as hollow circles
    sns.boxplot(ax=ax, data=reordered, color="white", width=0.5,
                showfliers=True, zorder=2,
                boxprops=dict(linewidth=0.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                medianprops=dict(color="black", linewidth=1.2),
                flierprops=dict(marker='o', markerfacecolor='none',
                                markeredgecolor='black', markersize=5,
                                markeredgewidth=0.8))

    # [Note: Section 2 (Strip Plot) and Section 3 (Connecting Lines) have been removed]

    # 4. Setup Grid (Horizontal Only)
    ax.minorticks_on()
    ax.grid(True, axis='y', which="major", color="0.80", linestyle='-')
    ax.grid(True, axis='y', which="minor", color="0.85", linestyle=(0, (1.2, 4.5)))
    ax.grid(False, axis='x')
    ax.set_box_aspect(1)

    # 5. Formatting
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Condition")
    ax.set_ylabel(unit, fontsize=13)

    # 6. Save and Show
    filename = f"{name.replace(' ', '_')}_results.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()