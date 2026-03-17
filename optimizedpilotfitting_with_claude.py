import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.optimize as opt
import os
from Datasetcode import dataset

# ==============================================================================
# OUTPUT DIRECTORY
# ==============================================================================

os.makedirs("FIGURES/fits", exist_ok=True)

# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================

costs = []

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
# COST FUNCTION
# ==============================================================================

def cost_function(params, w, vis_data, vest_data, condition, weight_vis=1.0, weight_vest=1.0):
    """
    Joint cost function over visual and vestibular models.

    Parameter vector layout depends on condition:
      - No-motion (C1-C3): only visual params are meaningful.
        params = [Kp, TL, TI, tau, omega_nm_vis, zeta_nm_vis]
      - Motion (C4-C6): visual + vestibular params fitted together.
        params = [Kp, TL, TI, tau, omega_nm_vis, zeta_nm_vis,
                  Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm_vest, zeta_nm_vest]

    Neuromuscular parameters are fitted independently for each model.
    Cost is a normalised sum of squared errors in the complex frequency domain.
    """
    # Guard: reject unphysical parameters early
    if condition not in [4, 5, 6]:
        Kp, TL, TI, tau, omega_nm_vis, zeta_nm_vis = params[:6]
        if omega_nm_vis <= 0 or zeta_nm_vis <= 0 or TI <= 0 or tau < 0:
            return 1e10
    else:
        Kp, TL, TI, tau, omega_nm_vis, zeta_nm_vis = params[:6]
        Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm_vest, zeta_nm_vest = params[6:]
        if (omega_nm_vis <= 0 or zeta_nm_vis <= 0 or TI <= 0 or tau < 0 or
                omega_nm_vest <= 0 or zeta_nm_vest <= 0 or tau_m < 0 or
                Tsc2 <= 0 or Tsc3 <= 0):
            return 1e10

    # --- Visual params (always present) ---
    Kp, TL, TI, tau, omega_nm_vis, zeta_nm_vis = params[:6]

    vis_model = Hpe_model(w, Kp, TL, TI, tau, omega_nm_vis, zeta_nm_vis)
    err_vis = np.abs(vis_data - vis_model)**2 / (np.abs(vis_data)**2 + 1e-12)
    cost = weight_vis * np.sum(err_vis)

    # --- Vestibular params (motion conditions only) ---
    if condition in [4, 5, 6]:
        Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm_vest, zeta_nm_vest = params[6:]
        vest_model = Hpxd_model(w, Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm_vest, zeta_nm_vest)
        err_vest = np.abs(vest_data - vest_model)**2 / (np.abs(vest_data)**2 + 1e-12)
        cost += weight_vest * np.sum(err_vest)

    return float(np.real(cost))


# ==============================================================================
# PARAMETER BOUNDS & INITIAL GUESSES
# ==============================================================================

# Visual (Hpe) parameter bounds:  Kp, TL, TI, tau, omega_nm, zeta_nm
VIS_BOUNDS = [
    (0.01, 20.0),   # Kp
    (0.0,  2.0),    # TL
    (0.01, 5.0),    # TI
    (0.01, 0.5),    # tau  [s]
    (5.0,  25.0),   # omega_nm  [rad/s]
    (0.1,  1.0),    # zeta_nm
]

# Vestibular (Hpxd) parameter bounds: Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm, zeta_nm
VEST_BOUNDS = [
    (-5.0,  5.0),   # Km  (can be negative — motion opposition)
    (0.0,   5.0),   # Tsc1
    (0.01,  5.0),   # Tsc2
    (0.01,  5.0),   # Tsc3
    (0.01,  0.5),   # tau_m  [s]
    (5.0,  25.0),   # omega_nm  [rad/s]
    (0.1,   1.0),   # zeta_nm
]

# Initial guess — centre of bounds
def _mid(bounds):
    return [(lo + hi) / 2 for lo, hi in bounds]

VIS_X0   = _mid(VIS_BOUNDS)
VEST_X0  = _mid(VEST_BOUNDS)

# Number of random multi-start restarts
N_STARTS = 8
RNG = np.random.default_rng(42)


def random_x0(bounds):
    """Draw a random initial point uniformly within bounds."""
    return [RNG.uniform(lo, hi) for lo, hi in bounds]


# ==============================================================================
# OPTIMISATION
# ==============================================================================

def fit_subject_condition(subject, condition, verbose=True):
    """
    Fit the pilot model for one subject / condition combination.

    Returns
    -------
    result : dict with keys
        'params'     – best-fit parameter array
        'cost'       – final cost value
        'vis_fit'    – complex Hpe model evaluated at w_FC
        'vest_fit'   – complex Hpxd model evaluated at w_FC  (None for C1-C3)
        'w'          – frequency vector
        'vis_data'   – measured Hpe_FC
        'vest_data'  – measured Hpxd_FC
        'subject'    – subject index
        'condition'  – condition index
    """
    rec = dataset[subject][condition]
    w         = rec["w_FC"].ravel().astype(float)
    vis_data  = rec["Hpe_FC"].ravel()
    vest_data = rec["Hpxd_FC"].ravel()

    motion = condition in [4, 5, 6]

    if motion:
        bounds = VIS_BOUNDS + VEST_BOUNDS
        x0_base = VIS_X0 + VEST_X0
    else:
        bounds = VIS_BOUNDS
        x0_base = VIS_X0

    best_cost   = np.inf
    best_params = None

    starts = [x0_base] + [random_x0(bounds) for _ in range(N_STARTS - 1)]

    for x0 in starts:
        try:
            res = opt.minimize(
                cost_function,
                x0,
                args=(w, vis_data, vest_data, condition),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
            )
            if res.fun < best_cost:
                best_cost   = res.fun
                best_params = res.x
        except Exception:
            continue
    
    costs.append(best_cost)

    if best_params is None:
        raise RuntimeError(f"Optimisation failed for subject {subject}, condition {condition}")

    # Evaluate fitted models
    Kp, TL, TI, tau, omega_nm_vis, zeta_nm_vis = best_params[:6]
    vis_fit = Hpe_model(w, Kp, TL, TI, tau, omega_nm_vis, zeta_nm_vis)

    if motion:
        Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm_vest, zeta_nm_vest = best_params[6:]
        vest_fit = Hpxd_model(w, Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm_vest, zeta_nm_vest)
    else:
        vest_fit = None

    if verbose:
        print(f"  Subj {subject}, Cond {condition}  →  cost = {best_cost:.4f}")

    return {
        "params":    best_params,
        "cost":      best_cost,
        "vis_fit":   vis_fit,
        "vest_fit":  vest_fit,
        "w":         w,
        "vis_data":  vis_data,
        "vest_data": vest_data,
        "subject":   subject,
        "condition": condition,
    }


# ==============================================================================
# PLOTTING
# ==============================================================================

CONDITION_LABELS = {
    1: "C1 – Gain, no motion",
    2: "C2 – Integrator, no motion",
    3: "C3 – Double-int, no motion",
    4: "C4 – Gain, motion",
    5: "C5 – Integrator, motion",
    6: "C6 – Double-int, motion",
}

COLOR_DATA = "#2c7bb6"
COLOR_FIT  = "#d7191c"


def _bode_arrays(H, w_dense=None):
    """Return (magnitude_dB, phase_deg_unwrapped) for a transfer function array."""
    mag_db = 20 * np.log10(np.abs(H) + 1e-30)
    phase  = np.angle(H, deg=True)
    phase  = np.unwrap(phase, period=360)
    return mag_db, phase


def plot_fit(result, save=True, show=False):
    """
    Bode plot comparing measured data with fitted model.
    Two rows (magnitude / phase), two columns (Hpe / Hpxd).
    """
    subj      = result["subject"]
    cond      = result["condition"]
    motion    = cond in [4, 5, 6]
    w         = result["w"]
    w_dense   = np.logspace(np.log10(w.min() * 0.8), np.log10(w.max() * 1.2), 300)

    # Re-evaluate fitted model on dense grid for smooth curves
    params = result["params"]
    Kp, TL, TI, tau, omega_nm_vis, zeta_nm_vis = params[:6]
    vis_dense = Hpe_model(w_dense, Kp, TL, TI, tau, omega_nm_vis, zeta_nm_vis)

    if motion:
        Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm_vest, zeta_nm_vest = params[6:]
        vest_dense = Hpxd_model(w_dense, Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm_vest, zeta_nm_vest)
    else:
        vest_dense = None

    # --- Measured data ---
    vis_mag_d,  vis_ph_d  = _bode_arrays(result["vis_data"])
    vest_mag_d, vest_ph_d = _bode_arrays(result["vest_data"])

    # --- Fitted data (discrete points) ---
    vis_mag_f,  vis_ph_f  = _bode_arrays(result["vis_fit"])
    vest_mag_f, vest_ph_f = _bode_arrays(result["vest_fit"]) if motion else (None, None)

    # --- Fitted smooth curves ---
    vis_mag_s,  vis_ph_s  = _bode_arrays(vis_dense)
    vest_mag_s, vest_ph_s = _bode_arrays(vest_dense) if motion else (None, None)

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        f"Subject {subj}  –  {CONDITION_LABELS[cond]}\n"
        f"Cost = {result['cost']:.4f}",
        fontsize=13, fontweight="bold",
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    ax_vis_mag  = fig.add_subplot(gs[0, 0])
    ax_vis_ph   = fig.add_subplot(gs[1, 0])
    ax_vest_mag = fig.add_subplot(gs[0, 1])
    ax_vest_ph  = fig.add_subplot(gs[1, 1])

    # ---- Hpe (visual) ----
    ax_vis_mag.semilogx(w, vis_mag_d, 'o', color=COLOR_DATA, ms=6,
                        label="Measured", zorder=3)
    ax_vis_mag.semilogx(w_dense, vis_mag_s, '-', color=COLOR_FIT, lw=2,
                        label="Fit", zorder=2)
    ax_vis_mag.set_ylabel("Magnitude [dB]")
    ax_vis_mag.set_title("$H_{pe}$ (Visual)")
    ax_vis_mag.grid(True, which="both", ls="--", alpha=0.5)
    ax_vis_mag.legend(fontsize=9)

    ax_vis_ph.semilogx(w, vis_ph_d, 'o', color=COLOR_DATA, ms=6, zorder=3)
    ax_vis_ph.semilogx(w_dense, vis_ph_s, '-', color=COLOR_FIT, lw=2, zorder=2)
    ax_vis_ph.set_xlabel("Frequency [rad/s]")
    ax_vis_ph.set_ylabel("Phase [deg]")
    ax_vis_ph.grid(True, which="both", ls="--", alpha=0.5)

    # ---- Hpxd (vestibular) ----
    ax_vest_mag.semilogx(w, vest_mag_d, 's', color=COLOR_DATA, ms=6,
                         label="Measured", zorder=3)
    if motion:
        ax_vest_mag.semilogx(w_dense, vest_mag_s, '-', color=COLOR_FIT, lw=2,
                             label="Fit", zorder=2)
    ax_vest_mag.set_title("$H_{pxd}$ (Vestibular)")
    ax_vest_mag.grid(True, which="both", ls="--", alpha=0.5)
    ax_vest_mag.legend(fontsize=9)
    if not motion:
        ax_vest_mag.text(0.5, 0.5, "No vestibular fit\n(no-motion condition)",
                         ha="center", va="center", transform=ax_vest_mag.transAxes,
                         fontsize=10, color="grey", style="italic")

    ax_vest_ph.semilogx(w, vest_ph_d, 's', color=COLOR_DATA, ms=6, zorder=3)
    if motion:
        ax_vest_ph.semilogx(w_dense, vest_ph_s, '-', color=COLOR_FIT, lw=2, zorder=2)
    ax_vest_ph.set_xlabel("Frequency [rad/s]")
    ax_vest_ph.grid(True, which="both", ls="--", alpha=0.5)
    if not motion:
        ax_vest_ph.text(0.5, 0.5, "No vestibular fit\n(no-motion condition)",
                        ha="center", va="center", transform=ax_vest_ph.transAxes,
                        fontsize=10, color="grey", style="italic")

    for ax in [ax_vis_mag, ax_vest_mag]:
        ax.set_ylabel("Magnitude [dB]")
    for ax in [ax_vis_ph, ax_vest_ph]:
        ax.set_ylabel("Phase [deg]")

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save:
        path = f"FIGURES/fits/Subj{subj}_Cond{cond}_fit.png"
        fig.savefig(path, dpi=150)
        print(f"    Saved → {path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_parameter_summary(all_results, save=True, show=False):
    """
    Bar charts of fitted visual parameters across conditions and subjects.
    One figure per parameter (Kp, TL, TI, tau, omega_nm, zeta_nm).
    """
    param_names = ["Kp", "TL", "TI", "tau", "omega_nm_vis", "zeta_nm_vis"]
    n_params    = len(param_names)

    subjects   = sorted({r["subject"]   for r in all_results})
    conditions = sorted({r["condition"] for r in all_results})
    n_subj = len(subjects)
    n_cond = len(conditions)

    # Build array: params_arr[subj_idx, cond_idx, param_idx]
    params_arr = np.full((n_subj, n_cond, n_params), np.nan)
    for r in all_results:
        si = subjects.index(r["subject"])
        ci = conditions.index(r["condition"])
        params_arr[si, ci, :] = r["params"][:n_params]

    cmap   = plt.cm.get_cmap("tab10", n_cond)
    x      = np.arange(n_subj)
    width  = 0.12

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Fitted Visual Parameters – All Subjects & Conditions", fontsize=13, fontweight="bold")

    for pi, (ax, pname) in enumerate(zip(axes.ravel(), param_names)):
        for ci, cond in enumerate(conditions):
            offset = (ci - n_cond / 2 + 0.5) * width
            ax.bar(x + offset, params_arr[:, ci, pi],
                   width=width, color=cmap(ci),
                   label=CONDITION_LABELS[cond] if pi == 0 else "_")
        ax.set_title(pname, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}" for s in subjects])
        ax.set_xlabel("Subject")
        ax.grid(axis="y", ls="--", alpha=0.4)

    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(ci))
               for ci, _ in enumerate(conditions)]
    labels  = [CONDITION_LABELS[c] for c in conditions]
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    if save:
        path = "FIGURES/fits/parameter_summary.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"    Saved → {path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_cost_heatmap(all_results, save=True, show=False):
    """Heat-map of final optimisation cost per subject × condition."""
    subjects   = sorted({r["subject"]   for r in all_results})
    conditions = sorted({r["condition"] for r in all_results})

    cost_mat = np.full((len(subjects), len(conditions)), np.nan)
    for r in all_results:
        si = subjects.index(r["subject"])
        ci = conditions.index(r["condition"])
        cost_mat[si, ci] = r["cost"]

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(cost_mat, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Normalised cost")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([f"Subject {s}" for s in subjects])
    ax.set_title("Optimisation Cost – Subject × Condition", fontsize=12, fontweight="bold")

    for si in range(len(subjects)):
        for ci in range(len(conditions)):
            ax.text(ci, si, f"{cost_mat[si, ci]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if cost_mat[si, ci] < cost_mat.max() * 0.6 else "white")

    plt.tight_layout()

    if save:
        path = "FIGURES/fits/cost_heatmap.png"
        fig.savefig(path, dpi=150)
        print(f"    Saved → {path}")

    if show:
        plt.show()
    plt.close(fig)


# ==============================================================================
# MAIN LOOP
# ==============================================================================

def print_params(result):
    """Pretty-print the fitted parameters."""
    p    = result["params"]
    cond = result["condition"]
    print(f"    Visual  → Kp={p[0]:.3f}, TL={p[1]:.3f}, TI={p[2]:.3f}, "
          f"tau={p[3]:.3f}, omega_nm={p[4]:.2f}, zeta_nm={p[5]:.3f}")
    if cond in [4, 5, 6]:
        print(f"    Vestib  → Km={p[6]:.3f}, Tsc1={p[7]:.3f}, Tsc2={p[8]:.3f}, "
              f"Tsc3={p[9]:.3f}, tau_m={p[10]:.3f}, omega_nm={p[11]:.2f}, zeta_nm={p[12]:.3f}")


if __name__ == "__main__":

    all_results = []

    subjects   = sorted(dataset.keys())
    conditions = sorted(next(iter(dataset.values())).keys())

    print("=" * 60)
    print("  Pilot model optimisation")
    print(f"  Subjects: {subjects}")
    print(f"  Conditions: {conditions}")
    print(f"  Multi-start restarts: {N_STARTS}")
    print("=" * 60)

    for subj in subjects:
        print(f"\nSubject {subj}")
        for cond in conditions:
            result = fit_subject_condition(subj, cond, verbose=True)
            print_params(result)
            all_results.append(result)
            plot_fit(result, save=True, show=False)

    print("\n" + "=" * 60)
    print("  Generating summary plots …")
    plot_parameter_summary(all_results, save=True, show=False)
    plot_cost_heatmap(all_results, save=True, show=False)

    print("\nDone. All figures saved to FIGURES/fits/")
    print("=" * 60)

    #Print costs summary
    npcosts = np.array(costs)
    print('mean:', np.mean(npcosts), '\n max:', np.max(npcosts), '\n min', np.min(npcosts))