import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
from Datasetcode import dataset


# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================

def Hnm_model(w, omega_nm, zeta_nm):
    """Neuromuscular actuation model."""
    s = 1j * w
    return omega_nm**2 / (s**2 + 2 * zeta_nm * omega_nm * s + omega_nm**2)


def Hpe_model(w, Kp, TL, TI, tau, omega_nm, zeta_nm):
    """
    Visual pilot model (Hpe).
    Params: Kp, TL, TI, tau, omega_nm, zeta_nm
    """
    s = 1j * w
    equalization = (TL * s + 1) / (TI * s + 1)
    nm = Hnm_model(w, omega_nm, zeta_nm)
    delay = np.exp(-s * tau)
    return Kp * equalization * nm * delay


def Hsc_model(w, Tsc1, Tsc2, Tsc3):
    """Semicircular canal dynamics model."""
    s = 1j * w
    return (1 + Tsc1 * s) / ((1 + Tsc2 * s) * (1 + Tsc3 * s))


def Hpxd_model(w, Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm, zeta_nm):
    """
    Vestibular pilot model (Hpxd).
    Params: Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm, zeta_nm
    """
    s = 1j * w
    sc = Hsc_model(w, Tsc1, Tsc2, Tsc3)
    nm = Hnm_model(w, omega_nm, zeta_nm)
    delay = np.exp(-s * tau_m)
    return Km * s * sc * delay * nm


# ==============================================================================
# FITTING HELPERS
# ==============================================================================

def prepare_fit_inputs(subject, condition):
    """Load and flatten the data used during fitting once per subject/condition."""
    data = dataset[subject][condition]
    w = data["w_FC"].flatten().astype(float)
    vis_data = data["Hpe_FC"].flatten()
    vest_data = data["Hpxd_FC"].flatten()

    return {
        "w": w,
        "vis_data": vis_data,
        "vest_data": vest_data,
        "vis_scale": np.sqrt(np.abs(vis_data) ** 2 + 1e-12),
        "vest_scale": np.sqrt(np.abs(vest_data) ** 2 + 1e-12),
        "condition": condition,
        "is_motion": condition in [4, 5, 6],
    }


def residual_vector(params, fit_input, weight_vis=1.0, weight_vest=1.0):
    """
    Weighted residual vector over visual and vestibular complex frequency data.

    The optimizer works on stacked real and imaginary residuals so the local
    least-squares step can use an algorithm tailored to residual minimization.
    """
    w = fit_input["w"]
    vis_data = fit_input["vis_data"]

    Kp, TL, TI, tau, omega_nm_vis, zeta_nm_vis = params[:6]

    vis_model = Hpe_model(w, Kp, TL, TI, tau, omega_nm_vis, zeta_nm_vis)
    vis_residual = np.sqrt(weight_vis) * (vis_data - vis_model) / fit_input["vis_scale"]
    residuals = [vis_residual.real, vis_residual.imag]

    if fit_input["is_motion"]:
        vest_data = fit_input["vest_data"]
        Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm_vest, zeta_nm_vest = params[6:]
        vest_model = Hpxd_model(w, Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm_vest, zeta_nm_vest)
        vest_residual = np.sqrt(weight_vest) * (vest_data - vest_model) / fit_input["vest_scale"]
        residuals.extend([vest_residual.real, vest_residual.imag])

    return np.concatenate(residuals)


def cost_function(params, fit_input, weight_vis=1.0, weight_vest=1.0):
    """Scalar objective used by global optimizers."""
    residuals = residual_vector(params, fit_input, weight_vis=weight_vis, weight_vest=weight_vest)
    return float(np.dot(residuals, residuals))


# ==============================================================================
# INITIAL GUESSES AND BOUNDS
# ==============================================================================

# Visual model: [Kp, TL, TI, tau, omega_nm, zeta_nm]
VIS_X0     = [1.0,  0.5,  1.0,  0.2,  10.0, 0.3]
VIS_BOUNDS = ([0.01, 0.0,  0.01, 0.01,  1.0, 0.1],
              [50.0, 10.0, 50.0, 1.0,  30.0, 1.0])

# Vestibular model: [Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm, zeta_nm]
VEST_X0     = [1.0,  5.0,  5.0,  5.0,  0.2,  10.0, 0.3]
VEST_BOUNDS = ([0.001, 0.1, 0.1, 0.1, 0.01,  1.0, 0.1],
               [50.0, 50.0, 50.0, 50.0, 1.0, 30.0, 1.0])

CONDITION_NAMES = {
    1: "Gain (P), no motion",
    2: "Integrator (V), no motion",
    3: "Double integrator (A), no motion",
    4: "Gain (P), motion",
    5: "Integrator (V), motion",
    6: "Double integrator (A), motion",
}


# ==============================================================================
# FITTING FUNCTION
# ==============================================================================

def fit_subject_condition(subject, condition, n_restarts=5):
    """
    Fit the pilot model for one subject/condition using differential evolution
    followed by a local L-BFGS-B polish. Multiple random restarts are used to
    reduce the risk of local minima.

    Returns a dict with fitted parameters and the final cost value.
    """
    fit_input = prepare_fit_inputs(subject, condition)
    is_motion = fit_input["is_motion"]

    if is_motion:
        x0     = VIS_X0 + VEST_X0
        bounds = list(zip(VIS_BOUNDS[0] + VEST_BOUNDS[0],
                          VIS_BOUNDS[1] + VEST_BOUNDS[1]))
    else:
        x0     = VIS_X0
        bounds = list(zip(VIS_BOUNDS[0], VIS_BOUNDS[1]))

    best_result = None
    best_cost   = np.inf

    # --- Global search with differential evolution ---
    de_result = opt.differential_evolution(
        cost_function,
        bounds=bounds,
        args=(fit_input,),
        seed=42,
        maxiter=300,
        tol=1e-6,
        popsize=10,
        mutation=(0.5, 1.5),
        recombination=0.7,
        polish=False,
    )

    if de_result.fun < best_cost:
        best_cost   = de_result.fun
        best_result = de_result

    local_starts = [np.array(de_result.x, dtype=float), np.array(x0, dtype=float)]
    for k in range(n_restarts):
        rng = np.random.default_rng(seed=k * 7)
        local_starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float))

    lower_bounds = np.array([lo for lo, _ in bounds], dtype=float)
    upper_bounds = np.array([hi for _, hi in bounds], dtype=float)

    # --- Local refinement from multiple starting points ---
    for x0_k in local_starts:
        x0_k = np.clip(x0_k, lower_bounds, upper_bounds)
        res = opt.least_squares(
            residual_vector,
            x0_k,
            args=(fit_input,),
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            max_nfev=4000,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        res.fun = float(np.dot(res.fun, res.fun))

        if res.fun < best_cost:
            best_cost   = res.fun
            best_result = res

    params = best_result.x

    # --- Unpack results ---
    result = {
        "Kp":         params[0],
        "TL":         params[1],
        "TI":         params[2],
        "tau":        params[3],
        "omega_nm_vis": params[4],
        "zeta_nm_vis":  params[5],
        "cost":       best_cost,
    }

    if is_motion:
        result.update({
            "Km":           params[6],
            "Tsc1":         params[7],
            "Tsc2":         params[8],
            "Tsc3":         params[9],
            "tau_m":        params[10],
            "omega_nm_vest": params[11],
            "zeta_nm_vest":  params[12],
        })

    return result


# ==============================================================================
# PLOTTING: BODE WITH MODEL OVERLAY
# ==============================================================================

def plot_fit(subject, condition, fit_params, save_dir="FIGURES_FIT"):
    """
    Plot measured Bode data with fitted model curves overlaid.
    Saves the figure to save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)

    data      = dataset[subject][condition]
    w_data    = data["w_FC"].flatten().astype(float)
    vis_data  = data["Hpe_FC"].flatten()
    vest_data = data["Hpxd_FC"].flatten()

    is_motion = condition in [4, 5, 6]

    # Evaluate fitted models on a dense frequency grid for smooth curves
    w_fine = np.logspace(np.log10(w_data.min()), np.log10(w_data.max()), 300)

    vis_model_fine = Hpe_model(
        w_fine,
        fit_params["Kp"], fit_params["TL"], fit_params["TI"],
        fit_params["tau"], fit_params["omega_nm_vis"], fit_params["zeta_nm_vis"]
    )

    # --- Magnitude & phase helpers ---
    def to_db(H):
        return 20 * np.log10(np.abs(H) + 1e-12)

    def to_phase(H):
        return np.unwrap(np.angle(H, deg=False)) * 180 / np.pi

    # Data
    vis_db_data   = to_db(vis_data)
    vis_ph_data   = np.unwrap(np.angle(vis_data, deg=True), period=360)
    vest_db_data  = to_db(vest_data)
    vest_ph_data  = np.unwrap(np.angle(vest_data, deg=True), period=360)

    # Model
    vis_db_model  = to_db(vis_model_fine)
    vis_ph_model  = to_phase(vis_model_fine)

    fig, axes = plt.subplots(2, 2 if is_motion else 1, figsize=(14 if is_motion else 7, 8))

    cond_name = CONDITION_NAMES.get(condition, f"C{condition}")
    fig.suptitle(f"Subject {subject} — Condition {condition}: {cond_name}\n"
                 f"Cost = {fit_params['cost']:.4f}", fontsize=12)

    # Collapse axes for no-motion case so indexing is uniform
    if not is_motion:
        axes = axes.reshape(2, 1)

    # ---- Visual column ----
    axes[0, 0].semilogx(w_data, vis_db_data,  'o', color='tab:blue',  label="Hpe data",  zorder=3)
    axes[0, 0].semilogx(w_fine, vis_db_model, '-', color='tab:blue',  label="Hpe model", linewidth=2)
    axes[0, 0].set_ylabel("Magnitude [dB]")
    axes[0, 0].set_title("Visual (Hpe)")
    axes[0, 0].grid(True, which="both", alpha=0.4)
    axes[0, 0].legend()

    axes[1, 0].semilogx(w_data, vis_ph_data,  'o', color='tab:blue',  label="Hpe data",  zorder=3)
    axes[1, 0].semilogx(w_fine, vis_ph_model, '-', color='tab:blue',  label="Hpe model", linewidth=2)
    axes[1, 0].set_ylabel("Phase [deg]")
    axes[1, 0].set_xlabel("Frequency [rad/s]")
    axes[1, 0].grid(True, which="both", alpha=0.4)
    axes[1, 0].legend()

    # ---- Vestibular column (motion conditions only) ----
    if is_motion:
        vest_model_fine = Hpxd_model(
            w_fine,
            fit_params["Km"], fit_params["Tsc1"], fit_params["Tsc2"], fit_params["Tsc3"],
            fit_params["tau_m"], fit_params["omega_nm_vest"], fit_params["zeta_nm_vest"]
        )
        vest_db_model = to_db(vest_model_fine)
        vest_ph_model = to_phase(vest_model_fine)

        axes[0, 1].semilogx(w_data, vest_db_data,  's', color='tab:orange', label="Hpxd data",  zorder=3)
        axes[0, 1].semilogx(w_fine, vest_db_model, '-', color='tab:orange', label="Hpxd model", linewidth=2)
        axes[0, 1].set_ylabel("Magnitude [dB]")
        axes[0, 1].set_title("Vestibular (Hpxd)")
        axes[0, 1].grid(True, which="both", alpha=0.4)
        axes[0, 1].legend()

        axes[1, 1].semilogx(w_data, vest_ph_data,  's', color='tab:orange', label="Hpxd data",  zorder=3)
        axes[1, 1].semilogx(w_fine, vest_ph_model, '-', color='tab:orange', label="Hpxd model", linewidth=2)
        axes[1, 1].set_ylabel("Phase [deg]")
        axes[1, 1].set_xlabel("Frequency [rad/s]")
        axes[1, 1].grid(True, which="both", alpha=0.4)
        axes[1, 1].legend()

    plt.tight_layout()
    filename = os.path.join(save_dir, f"Subject{subject}_Condition{condition}_fit.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"    Saved: {filename}")


# ==============================================================================
# PRINT RESULTS
# ==============================================================================

def print_results(subject, condition, fit_params):
    """Pretty-print the fitted parameters for one subject/condition."""
    is_motion  = condition in [4, 5, 6]
    cond_name  = CONDITION_NAMES.get(condition, f"C{condition}")

    print(f"\n  Subject {subject} | Condition {condition} ({cond_name})")
    print(f"  {'─'*50}")
    print(f"  Cost (J)    = {fit_params['cost']:.6f}")
    print(f"  --- Visual model (Hpe) ---")
    print(f"  Kp          = {fit_params['Kp']:.4f}")
    print(f"  TL          = {fit_params['TL']:.4f}")
    print(f"  TI          = {fit_params['TI']:.4f}")
    print(f"  tau         = {fit_params['tau']:.4f}  s")
    print(f"  omega_nm    = {fit_params['omega_nm_vis']:.4f}  rad/s")
    print(f"  zeta_nm     = {fit_params['zeta_nm_vis']:.4f}")
    if is_motion:
        print(f"  --- Vestibular model (Hpxd) ---")
        print(f"  Km          = {fit_params['Km']:.4f}")
        print(f"  Tsc1        = {fit_params['Tsc1']:.4f}  s")
        print(f"  Tsc2        = {fit_params['Tsc2']:.4f}  s")
        print(f"  Tsc3        = {fit_params['Tsc3']:.4f}  s")
        print(f"  tau_m       = {fit_params['tau_m']:.4f}  s")
        print(f"  omega_nm    = {fit_params['omega_nm_vest']:.4f}  rad/s")
        print(f"  zeta_nm     = {fit_params['zeta_nm_vest']:.4f}")


# ==============================================================================
# MAIN LOOP: ALL SUBJECTS × ALL CONDITIONS
# ==============================================================================

if __name__ == "__main__":

    N_SUBJECTS   = 6
    N_CONDITIONS = 6
    SAVE_DIR     = "FIGURES_FIT"

    # Store all results: all_results[subject][condition] = fit_params dict
    all_results = {}

    for subj in range(1, N_SUBJECTS + 1):
        all_results[subj] = {}
        print(f"\n{'='*60}")
        print(f"  SUBJECT {subj}")
        print(f"{'='*60}")

        for cond in range(1, N_CONDITIONS + 1):
            print(f"\n  Fitting Subject {subj}, Condition {cond}...", end=" ", flush=True)

            fit_params = fit_subject_condition(subj, cond, n_restarts=5)
            all_results[subj][cond] = fit_params

            print(f"done  (J = {fit_params['cost']:.4f})")

            # Print parameters to console
            print_results(subj, cond, fit_params)

            # Save Bode plot with model overlay
            plot_fit(subj, cond, fit_params, save_dir=SAVE_DIR)

    print(f"\n{'='*60}")
    print("  All fits complete. Figures saved to:", SAVE_DIR)
    print(f"{'='*60}\n")
