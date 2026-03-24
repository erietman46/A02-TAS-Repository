import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
from Datasetcode import dataset
import scipy.optimize as opt
from scipy.stats import qmc  # <-- ADD THIS

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


def Hpxd_model(w, omega_nm, zeta_nm, Km, Tsc1, Tsc2, Tsc3, tau_m):
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
    # --- Visual params (always present) ---
    Kp, TL, TI, tau, omega_nm, zeta_nm = params[:6]

    vis_model = Hpe_model(w, Kp, TL, TI, tau, omega_nm, zeta_nm)
    err_vis = np.abs(vis_data - vis_model)**2 / (np.abs(vis_data)**2 + 1e-12)
    cost = weight_vis * np.sum(err_vis)

    # --- Vestibular params (motion conditions only) ---
    if condition in [4, 5, 6]:
        omega_nm, zeta_nm, Km, Tsc1, Tsc2, Tsc3, tau_m = params[4:]
        vest_model = Hpxd_model(w, omega_nm, zeta_nm, Km, Tsc1, Tsc2, Tsc3, tau_m )
        err_vest = np.abs(vest_data - vest_model)**2 / (np.abs(vest_data)**2 + 1e-12)
        cost += weight_vest * np.sum(err_vest)

    return float(np.real(cost))



# ==============================================================================
# PARAMETER BOUNDS & INITIAL GUESSES
# ==============================================================================

VIS_BOUNDS = [
    (0.01, 10.0),   # Kp
    (0.0,  10.0),    # TL
    (0.0, 10.0),    # TI
    (0.01, 0.5),    # tau  [s]
    (5.0,  35.0),   # omega_nm  [rad/s]
    (0.1,  1.0),    # zeta_nm
]

# Vestibular (Hpxd) parameter bounds: Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm, zeta_nm
VEST_BOUNDS = [
    (-5.0,  5.0),   # Km  (can be negative — motion opposition)
    (0.11,   0.11),   # Tsc1
    (5.924,  5.924),   # Tsc2
    (0.005,  0.005),   # Tsc3
    (0.01,  1.5),   # tau_m  [s]
]

def _mid(bounds):
    return np.array([(lo + hi) / 2 for lo, hi in bounds], dtype=float)


VIS_X0 = _mid(VIS_BOUNDS)
VEST_X0 = _mid(VEST_BOUNDS)

N_STARTS = 30  # Number of multistart initial points (including the mid-point)


def initialization(condition):
    """Return initial guess and bounds for a given condition."""
    motion = condition in [4, 5, 6]

    if motion:
        x0 = np.concatenate([VIS_X0, VEST_X0])
        bounds = VIS_BOUNDS + VEST_BOUNDS
    else:
        x0 = VIS_X0.copy()
        bounds = VIS_BOUNDS.copy()

    return x0, bounds

def parameter_names(condition):
    """Return ordered parameter names for a given condition."""
    base_names = ['Kp', 'TL', 'TI', 'tau', 'omega_nm', 'zeta_nm']
    if condition in [4, 5, 6]:
        base_names += ['Km', 'Tsc1', 'Tsc2', 'Tsc3', 'tau_m' ]
    return base_names

def clip_to_bounds(x, bounds):
    lo, hi = bounds_to_arrays(bounds)
    return np.minimum(np.maximum(x, lo), hi)

def bounds_to_arrays(bounds):
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo, hi

def lhs_points(bounds, n_points, seed):
    """
    LHS samples scaled to bounds, supporting fixed bounds (lo == hi).
    Returns array shape (n_points, dim).
    """
    lo, hi = bounds_to_arrays(bounds)
    dim = len(bounds)

    # Identify free vs fixed dimensions
    free_mask = hi > lo
    fixed_mask = ~free_mask

    X = np.zeros((n_points, dim), dtype=float)

    # Fill fixed dimensions directly
    if np.any(fixed_mask):
        X[:, fixed_mask] = lo[fixed_mask]  # same as hi

    # LHS only on free dimensions
    n_free = int(np.sum(free_mask))
    if n_free > 0:
        sampler = qmc.LatinHypercube(d=n_free, seed=int(seed))
        u = sampler.random(n=n_points)  # [0,1]
        lo_f = lo[free_mask]
        hi_f = hi[free_mask]
        # Manual scaling (works because all hi_f > lo_f)
        X[:, free_mask] = lo_f + u * (hi_f - lo_f)

    return X

# ==============================================================================
# FITTING
# ==============================================================================

def fit_subject_condition(subject, condition, weight_vis=1.0, weight_vest=1.0, verbose=True):
    """
    Fit model parameters for a given subject and condition using multistart optimization.

    Returns
    -------
    visual_fit : np.ndarray
        Fitted visual model frequency response.
    vest_fit : np.ndarray or None
        Fitted vestibular model frequency response for motion conditions.
    result : OptimizeResult
        Best scipy optimization result.
    final_cost : float
        Final value of the objective function.
    """
    motion = condition in [4, 5, 6]

    try:
        rec = dataset[subject][condition]
    except KeyError as e:
        raise KeyError(f"Missing dataset entry for subject={subject}, condition={condition}") from e

    w = np.asarray(rec["w_FC"]).ravel().astype(float)
    vis_data = np.asarray(rec["Hpe_FC"]).ravel()
    vest_data = np.asarray(rec["Hpxd_FC"]).ravel() if motion else None

    x0_base, bounds = initialization(condition)

    best_cost = np.inf
    best_params = None
    best_result = None

    # --- Multistart initialization using LHS ---
    # Deterministic seed per (subject, condition) so that parallel runs reproduce exactly.
    lhs_seed = 10_000 * int(subject) + int(condition)

    if N_STARTS > 1:
        X_lhs = lhs_points(bounds, n_points=N_STARTS - 1, seed=lhs_seed)
        starts = [x0_base] + [X_lhs[k, :] for k in range(X_lhs.shape[0])]
    else:
        starts = [x0_base]

    for x0 in starts:
        try:
            res = opt.minimize(
                cost_function,
                x0=np.asarray(x0, dtype=float),
                args=(w, vis_data, vest_data, condition, weight_vis, weight_vest),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
            )

            if np.isfinite(res.fun) and res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x
                best_result = res

        except Exception:
            continue

    if best_result is None:
        raise RuntimeError(f"Optimisation failed for subject {subject}, condition {condition}")

    fitted_params = best_params

    visual_fit = Hpe_model(w, *fitted_params[:6])

    # NOTE: This is only correct if your motion parameter vector matches what Hpxd_model expects.
    # Make sure your motion parameter ordering/slicing is fixed (we discussed the bug earlier).
    vest_fit = Hpxd_model(w, *fitted_params[4:]) if motion else None

    names = parameter_names(condition)
    dataset[subject][condition]["fitted_params"] = dict(zip(names, fitted_params))
    dataset[subject][condition]["fit_cost"] = best_cost
    dataset[subject][condition]["fit_success"] = best_result.success
    dataset[subject][condition]["fit_message"] = best_result.message

    if verbose:
        print(f"  Subj {subject}, Cond {condition}  →  cost = {best_cost:.4f}")

    return visual_fit, vest_fit, best_result, best_cost



visual_fit, vest_fit, result, result.fun = fit_subject_condition(1,4)



def fmin_minimise(subject, condition):
    
    """
    Fit model parameters for a given subject and condition.

    Returns
    -------
    visual_fit : np.ndarray
        Fitted visual model frequency response.
    vest_fit : np.ndarray or None
        Fitted vestibular model frequency response for motion conditions.
    final_cost : float
        Final value of the objective function.
    result : OptimizeResult
        Full scipy optimization result.
    """
    motion = condition in [4, 5, 6]

    try:
        rec = dataset[subject][condition]
    except KeyError as e:
        raise KeyError(f"Missing dataset entry for subject={subject}, condition={condition}") from e

    w_FC = np.asarray(rec['w_FC'])
    vis_data = np.asarray(rec['Hpe_FC'])
    vest_data = np.asarray(rec['Hpxd_FC']) if motion else None

    x0, bounds = initialization(condition)
    
    return