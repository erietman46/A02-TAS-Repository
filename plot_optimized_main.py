import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, faster file saving, no GUI overhead
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

#plot style
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

#plot resolution
FIG_DPI = 220

def bode_mag_phase(H): 
    mag = np.abs(H)
    phase_deg = np.unwrap(np.angle(H)) * 180.0 / np.pi
    return mag, phase_deg

def setup_bode_axis(ax, ylabel: str, phase_plot: bool = False):
    ax.set_xscale("log")
    ax.minorticks_on()

    ax.grid(True, which="major", color="0.70")
    ax.grid(True, which="minor", color="0.70", linestyle=(0, (1.2, 4.5)))

    ax.set_xlabel(r"$\omega,\ \mathrm{rad\ s^{-1}}$")
    ax.set_ylabel(ylabel)
    ax.set_box_aspect(1)

    if phase_plot:
        ax.axhline(-180, color="0.4", linewidth=0.8)
    else:
        ax.axhline(1.0, color="0.4", linewidth=0.8)
        ax.set_yscale("log")


def run_one(i, j):
    from Datasetcode import dataset
    import Optimizedpilotfitting_main as opf

    motion = j in [4, 5, 6]

    w_FC = np.asarray(dataset[i][j]["w_FC"]).ravel()
    vis_data = np.asarray(dataset[i][j]["Hpe_FC"]).ravel()
    vest_data = np.asarray(dataset[i][j]["Hpxd_FC"]).ravel() if motion else None

    visual_fit, vestib_fit, best_result, best_cost = opf.fit_subject_condition(i, j, verbose=False)

    params = best_result.x.tolist()

    return i, j, motion, w_FC, vis_data, vest_data, visual_fit, vestib_fit, best_cost, params

#plotting
def save_visual_bode(i, j, w_FC, vis_data, visual_fit):
    vis_mag, vis_ang = bode_mag_phase(vis_data)
    fit_mag, fit_ang = bode_mag_phase(visual_fit)

    fig, axes = plt.subplots(2, 1, figsize=(5.4, 7.2))
    fig.subplots_adjust(hspace=0.42)

    ax = axes[0]
    setup_bode_axis(ax, r"$|H_{pe}|$", phase_plot=False)
    ax.plot(w_FC, fit_mag, "k-", linewidth=1.4, label="Fitted $H_{pe}$")
    ax.plot(w_FC, vis_mag, linestyle="none", marker=r"$\ast$", markersize=9, color="k", label="Measured $H_{pe}$")
    ax.relim()
    ax.autoscale_view()
    ax.set_title("a) Visual magnitude", y=-0.33, fontweight="bold")

    ax = axes[1]
    setup_bode_axis(ax, r"$\angle H_{pe},\ \mathrm{deg}$", phase_plot=True)
    ax.plot(w_FC, fit_ang, "k-", linewidth=1.4, label="Fitted $H_{pe}$")
    ax.plot(w_FC, vis_ang, linestyle="none", marker=r"$\ast$", markersize=9, color="k", label="Measured $H_{pe}$")
    ax.relim()
    ax.autoscale_view()
    ax.legend(
        loc="lower left",
        frameon=True,
        fancybox=False,
        edgecolor="k",
        borderpad=0.2,
        handlelength=1.0,
        handletextpad=0.35,
    )
    ax.set_title("b) Visual phase", y=-0.33, fontweight="bold")

    fig.suptitle(f"Subject {i} — Condition {j} — Visual ($H_{{pe}}$)", y=0.98, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"FIGURES/subject_{i}_condition_{j}_visual.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_vestibular_bode(i, j, w_FC, vest_data, vestib_fit):
    vest_mag, vest_ang = bode_mag_phase(vest_data)
    fit_mag, fit_ang = bode_mag_phase(vestib_fit)

    fig, axes = plt.subplots(2, 1, figsize=(5.4, 7.2))
    fig.subplots_adjust(hspace=0.42)

    ax = axes[0]
    setup_bode_axis(ax, r"$|H_{pxd}|$", phase_plot=False)
    ax.plot(w_FC, fit_mag, "k-", linewidth=1.4, label="Fitted $H_{pxd}$")
    ax.plot(w_FC, vest_mag, linestyle="none", marker=r"$\ast$", markersize=9, color="k", label="Measured $H_{pxd}$")
    ax.relim()
    ax.autoscale_view()
    ax.set_title("a) Vestibular magnitude", y=-0.33, fontweight="bold")

    ax = axes[1]
    setup_bode_axis(ax, r"$\angle H_{pxd},\ \mathrm{deg}$", phase_plot=True)
    ax.plot(w_FC, fit_ang, "k-", linewidth=1.4, label="Fitted $H_{pxd}$")
    ax.plot(w_FC, vest_ang, linestyle="none", marker=r"$\ast$", markersize=9, color="k", label="Measured $H_{pxd}$")
    ax.relim()
    ax.autoscale_view()
    ax.legend(
        loc="lower left",
        frameon=True,
        fancybox=False,
        edgecolor="k",
        borderpad=0.2,
        handlelength=1.0,
        handletextpad=0.35,
    )
    ax.set_title("b) Vestibular phase", y=-0.33, fontweight="bold")

    fig.suptitle(f"Subject {i} — Condition {j} — Vestibular ($H_{{pxd}}$)", y=0.98, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"FIGURES/subject_{i}_condition_{j}_vestibular.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    costs = []
    parameters_C1 = np.zeros((6, 11))
    parameters_C2 = np.zeros((6, 11))
    parameters_C3 = np.zeros((6, 11))
    parameters_C4 = np.zeros((6, 11))
    parameters_C5 = np.zeros((6, 11))
    parameters_C6 = np.zeros((6, 11))
    global_parameters = [parameters_C1, parameters_C2, parameters_C3,
                         parameters_C4, parameters_C5, parameters_C6]

    # Dispatch all 36 fits in parallel across available CPU cores
    futures = {}
    with ProcessPoolExecutor() as executor:
        for i in range(1, 7):
            for j in range(1, 7):
                futures[executor.submit(run_one, i, j)] = (i, j)

        for future in as_completed(futures):
            i, j, motion, w_FC, vis_data, vest_data, visual_fit, vestib_fit, cost, params = future.result()

            costs.append(cost)

            # Pad params to length 11
            params = list(params)
            while len(params) < 11:
                params.append(0.0)
            global_parameters[j - 1][i - 1] = params

            # Save figures in the main process 
            save_visual_bode(i, j, w_FC, vis_data, visual_fit)
            if motion:
                save_vestibular_bode(i, j, w_FC, vest_data, vestib_fit)

            print(f"Finished Subject {i}, Condition {j}, Cost = {cost:.4f}")

    # Print cost summary
    npcosts = np.array(costs)
    np.save("global_parameters.npy", np.array(global_parameters, dtype=object))

    print('mean:', np.mean(npcosts), '\n max:', np.max(npcosts), '\n min', np.min(npcosts))
