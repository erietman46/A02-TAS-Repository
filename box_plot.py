import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from error_pdf import error_pdf
from RMS_error import RMS_error
from derivative_e_RMS import RMS_DERe


def generate_individual_plots(data_dict, numbers, p_values, effect_sizes, units_dict):
    """
    Generates and saves a separate figure for each metric in the data_dict.
    """
    sns.set_theme(style="whitegrid")
    metrics = list(data_dict.keys())

    for metric in metrics:
        # Create a new figure for each metric
        plt.figure(figsize=(6, 6))

        no_mo = data_dict[metric][0]
        mo = data_dict[metric][1]
        plot_data = [no_mo, mo]
        labels = ['Fixed-Base', 'Motion-Base']

        # 1. Draw the Boxplot
        sns.boxplot(data=plot_data, palette="Pastel1", width=0.5, showfliers=False)

        # 2. Add Strip Plot (Individual dots)
        sns.stripplot(data=plot_data, color="black", size=7, jitter=False, alpha=0.8)

        # 3. Draw connecting lines between pairs
        for j in range(len(no_mo)):
            plt.plot([0, 1], [no_mo[j], mo[j]], color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # 4. Annotations (Significance and Effect Size)
        metric_label = numbers[metric]
        p_val = p_values[metric]
        es = effect_sizes[metric]
        stats_text = f"{metric_label}\np = {p_val:.4f}\nd = {es:.2f}"

        # Calculate text placement
        y_max = max(max(no_mo), max(mo))
        y_range = y_max - min(min(no_mo), min(mo))
        plt.text(0.5, y_max + (y_range * 0.1), stats_text,
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Formatting
        unit = units_dict.get(metric, "")
        #plt.title(f"Acceleration Task: {metric}", fontsize=14, fontweight='bold')
        plt.xticks([0, 1], labels)
        plt.ylabel(f"[{unit}]", fontsize=12)

        # Save each plot with a unique filename
        # We replace spaces with underscores for the filename
        filename = f"{metric.replace(' ', '_')}_accel_results.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved: {filename}")

        plt.show()  # Shows the window for the current metric before moving to the next


# --- EXAMPLE USAGE ---
# Data for the Acceleration condition only
my_units = {
    "RMS Error": "degrees",
    "RMS Error Deriv": "degrees/s",
    "1-Sigma Interval": "degrees"
}

def create_accel_data_dict(rms_err, rms_deriv, sigma_int):
    """
    Extracts the Acceleration columns (index 2 and 5)
    from the (6,6) metric arrays.
    """

    data_dict = {
        "RMS Error": [
            rms_err[:, 2],  # Column 2: Acceleration No-Motion
            rms_err[:, 5]  # Column 5: Acceleration Motion
        ],
        "RMS Error Deriv": [
            rms_deriv[:, 2],
            rms_deriv[:, 5]
        ],
        "1-Sigma Interval": [
            sigma_int[:, 2],
            sigma_int[:, 5]
        ]
    }

    return data_dict

_,_,_,_,sigma_interval_array = error_pdf()
_,_,rms_error_array = RMS_error()
_,_,rms_deriv_array = RMS_DERe()

accel_data = create_accel_data_dict(
    rms_error_array,
    rms_deriv_array,
    sigma_interval_array
)

numbers = {"RMS Error": "(a)", "RMS Error Deriv": "(b)", "1-Sigma Interval": "(c)"}
p_vals = {"RMS Error": 0.0033, "RMS Error Deriv": 0.0105, "1-Sigma Interval": 0.0012}
e_sizes = {"RMS Error": -2.14, "RMS Error Deriv": -1.63, "1-Sigma Interval": -2.72}

generate_individual_plots(accel_data, numbers, p_vals, e_sizes, my_units)