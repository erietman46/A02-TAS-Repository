import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — faster file saving, no GUI overhead
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


'''
TIME HISTORIES
• the error signal                  e [deg]   8192x5
• the control signal                u [deg]   8192x5
• the controlled yaw angle          x [deg]   8192x5
• the target signal                 ft [deg]   8192x1
• the disturbance signal            fd [deg]  8192x1
• the time vector                   t [s]   8192x1

MEASURED PILOT FREQUENCY RESPONSES
• the Hpe (visual) frequency response Hpe_FC [complex numbers] 20x1
• the Hpxd (motion) frequency response  Hpxd_FC [complex numbers] 20x1
• the frequency vector   w_FC [rad/s]   20x1
'''

'''
• C1 = Gain (P), no motion
• C2 = Single integrator (V), no motion
• C3 = Double integrator (A), no motion
• C4 = Gain (P), motion
• C5 = Single integrator (V), motion
• C6 = Double integrator (A), motion
'''

#__________________________________________________
## BODE PLOTS FOR PILOT RESPONSES
#__________________________________________________


def bode_mag_phase(H):
    """Return magnitude in dB and unwrapped phase in degrees."""
    H_abs = np.abs(H)
    H_db = 20 * np.log10(H_abs)
    H_ang = np.angle(H, deg=True)
    H_ang = np.unwrap(H_ang, period=360, axis=0)
    return H_db, H_ang


def run_one(i, j):
    """Run a single subject/condition fit. Executed in a worker process."""
    # Each worker process gets its own imports
    from Datasetcode import dataset
    import Optimizedpilotfitting_official_copy as opf

    motion = j in [4, 5, 6]

    w_FC = np.asarray(dataset[i][j]["w_FC"]).ravel()
    vis_data = np.asarray(dataset[i][j]["Hpe_FC"]).ravel()
    vest_data = np.asarray(dataset[i][j]["Hpxd_FC"]).ravel() if motion else None

    visual_fit, vestib_fit, best_result, best_cost = opf.fit_subject_condition(i, j, verbose=False)

    params = best_result.x.tolist()

    return i, j, motion, w_FC, vis_data, vest_data, visual_fit, vestib_fit, best_cost, params


def save_visual_bode(i, j, w_FC, vis_data, visual_fit):
    vis_db, vis_ang = bode_mag_phase(vis_data)
    visual_fit_db, visual_fit_ang = bode_mag_phase(visual_fit)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].semilogx(w_FC, vis_db, 'o', label="Measured Hpe")
    axes[0].semilogx(w_FC, visual_fit_db, '-', label="Fitted Hpe")
    axes[0].set_xlabel("Frequency [rad/s]")
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].set_title(f"Visual Bode Plot - Subject {i}, Condition {j}")
    axes[0].grid(True, which="both")
    axes[0].legend()

    axes[1].semilogx(w_FC, vis_ang, 'o', label="Measured Hpe")
    axes[1].semilogx(w_FC, visual_fit_ang, '-', label="Fitted Hpe")
    axes[1].set_xlabel("Frequency [rad/s]")
    axes[1].set_ylabel("Phase [deg]")
    axes[1].set_title(f"Visual Bode Plot - Subject {i}, Condition {j}")
    axes[1].grid(True, which="both")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(f"FIGURES/subject_{i}_condition_{j}_visual.png", dpi=200)
    plt.close(fig)


def save_vestibular_bode(i, j, w_FC, vest_data, vestib_fit):
    vest_db, vest_ang = bode_mag_phase(vest_data)
    vestib_fit_db, vest_fit_ang = bode_mag_phase(vestib_fit)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].semilogx(w_FC, vest_db, 's', label="Measured Hpxd")
    axes[0].semilogx(w_FC, vestib_fit_db, '-', label="Fitted Hpxd")
    axes[0].set_xlabel("Frequency [rad/s]")
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].set_title(f"Vestibular Bode Plot - Subject {i}, Condition {j}")
    axes[0].grid(True, which="both")
    axes[0].legend()

    axes[1].semilogx(w_FC, vest_ang, 's', label="Measured Hpxd")
    axes[1].semilogx(w_FC, vest_fit_ang, '-', label="Fitted Hpxd")
    axes[1].set_xlabel("Frequency [rad/s]")
    axes[1].set_ylabel("Phase [deg]")
    axes[1].set_title(f"Vestibular Bode Plot - Subject {i}, Condition {j}")
    axes[1].grid(True, which="both")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(f"FIGURES/subject_{i}_condition_{j}_vestibular.png", dpi=200)
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

            # Pad params to length 13
            params = list(params)
            while len(params) < 11:
                params.append(0.0)
            global_parameters[j - 1][i - 1] = params

            # Save figures in the main process (Agg backend is safe here)
            save_visual_bode(i, j, w_FC, vis_data, visual_fit)
            if motion:
                save_vestibular_bode(i, j, w_FC, vest_data, vestib_fit)

            print(f"Finished Subject {i}, Condition {j}, Cost = {cost:.4f}")

    # Print cost summary
    npcosts = np.array(costs)
    np.save("global_parameters.npy", np.array(global_parameters, dtype=object))
    
    print('mean:', np.mean(npcosts), '\n max:', np.max(npcosts), '\n min', np.min(npcosts))
    