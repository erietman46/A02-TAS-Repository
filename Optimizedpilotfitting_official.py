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

def _mid(bounds):
    return np.array([(lo + hi) / 2 for lo, hi in bounds], dtype=float)


VIS_X0 = _mid(VIS_BOUNDS)
VEST_X0 = _mid(VEST_BOUNDS)

N_STARTS = 30
RNG = np.random.default_rng(42)


def random_x0(bounds):
    """Draw a random initial point uniformly within bounds."""
    return np.array([RNG.uniform(lo, hi) for lo, hi in bounds], dtype=float)


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
    base_names = ['Kp', 'TL', 'TI', 'tau', 'omega_nm_vis', 'zeta_nm_vis']
    if condition in [4, 5, 6]:
        base_names += ['Km', 'Tsc1', 'Tsc2', 'Tsc3', 'tau_m', 'omega_nm_vest', 'zeta_nm_vest']
    return base_names


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

    starts = [x0_base] + [random_x0(bounds) for _ in range(N_STARTS - 1)]

    for x0 in starts:
        try:
            res = opt.minimize(
                cost_function,
                x0=x0,
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
    vest_fit = Hpxd_model(w, *fitted_params[6:]) if motion else None

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