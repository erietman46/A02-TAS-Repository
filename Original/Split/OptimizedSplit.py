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
# COST FUNCTIONS
# ==============================================================================

def visual_cost(params, w, vis_data):
    """Scalar cost for visual model only."""
    Kp, TL, TI, tau, omega_nm_vis, zeta_nm_vis = params
    vis_model = Hpe_model(w, Kp, TL, TI, tau, omega_nm_vis, zeta_nm_vis)
    err_vis = np.abs(vis_data - vis_model)**2 / (np.abs(vis_data)**2 + 1e-12)
    return float(np.real(np.sum(err_vis)))


def vestibular_cost(params, w, vest_data):
    """Scalar cost for vestibular model only."""
    Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm_vest, zeta_nm_vest = params
    vest_model = Hpxd_model(w, Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm_vest, zeta_nm_vest)
    err_vest = np.abs(vest_data - vest_model)**2 / (np.abs(vest_data)**2 + 1e-12)
    return float(np.real(np.sum(err_vest)))



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

def fit_subject_condition(subject, condition, verbose=True):
    """
    Fit visual and vestibular models separately for a given subject and condition.

    Returns
    -------
    visual_fit : np.ndarray
        Fitted visual model frequency response.
    vest_fit : np.ndarray or None
        Fitted vestibular model frequency response (None for no-motion conditions).
    visual_result : OptimizeResult
        Best optimization result for the visual model.
    vest_result : OptimizeResult or None
        Best optimization result for the vestibular model.
    """
    motion = condition in [4, 5, 6]

    try:
        rec = dataset[subject][condition]
    except KeyError as e:
        raise KeyError(f"Missing dataset entry for subject={subject}, condition={condition}") from e

    w = np.asarray(rec["w_FC"]).ravel().astype(float)
    vis_data = np.asarray(rec["Hpe_FC"]).ravel()
    vest_data = np.asarray(rec["Hpxd_FC"]).ravel() if motion else None

    # -----------------------------
    # Visual fit
    # -----------------------------
    best_vis_cost = np.inf
    best_vis_result = None
    best_vis_params = None

    vis_starts = [VIS_X0.copy()] + [random_x0(VIS_BOUNDS) for _ in range(N_STARTS - 1)]

    for x0 in vis_starts:
        try:
            res = opt.minimize(
                visual_cost,
                x0=x0,
                args=(w, vis_data),
                method="L-BFGS-B",
                bounds=VIS_BOUNDS,
                options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
            )

            if np.isfinite(res.fun) and res.fun < best_vis_cost:
                best_vis_cost = res.fun
                best_vis_result = res
                best_vis_params = res.x
        except Exception:
            continue

    if best_vis_result is None:
        raise RuntimeError(f"Visual optimisation failed for subject {subject}, condition {condition}")

    visual_fit = Hpe_model(w, *best_vis_params)

    # -----------------------------
    # Vestibular fit
    # -----------------------------
    best_vest_cost = None
    best_vest_result = None
    best_vest_params = None
    vest_fit = None

    if motion:
        best_vest_cost = np.inf
        vest_starts = [VEST_X0.copy()] + [random_x0(VEST_BOUNDS) for _ in range(N_STARTS - 1)]

        for x0 in vest_starts:
            try:
                res = opt.minimize(
                    vestibular_cost,
                    x0=x0,
                    args=(w, vest_data),
                    method="L-BFGS-B",
                    bounds=VEST_BOUNDS,
                    options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
                )

                if np.isfinite(res.fun) and res.fun < best_vest_cost:
                    best_vest_cost = res.fun
                    best_vest_result = res
                    best_vest_params = res.x
            except Exception:
                continue

        if best_vest_result is None:
            raise RuntimeError(f"Vestibular optimisation failed for subject {subject}, condition {condition}")

        vest_fit = Hpxd_model(w, *best_vest_params)

    # -----------------------------
    # Store results
    # -----------------------------
    rec["fitted_params_visual"] = dict(zip(
        ['Kp', 'TL', 'TI', 'tau', 'omega_nm_vis', 'zeta_nm_vis'],
        best_vis_params
    ))
    rec["fit_cost_visual"] = best_vis_cost
    rec["fit_success_visual"] = best_vis_result.success
    rec["fit_message_visual"] = best_vis_result.message

    if motion:
        rec["fitted_params_vestibular"] = dict(zip(
            ['Km', 'Tsc1', 'Tsc2', 'Tsc3', 'tau_m', 'omega_nm_vest', 'zeta_nm_vest'],
            best_vest_params
        ))
        rec["fit_cost_vestibular"] = best_vest_cost
        rec["fit_success_vestibular"] = best_vest_result.success
        rec["fit_message_vestibular"] = best_vest_result.message

    if verbose:
        print(f"Subj {subject}, Cond {condition}")
        print(f"  Visual cost     = {best_vis_cost:.4f}")
        if motion:
            print(f"  Vestibular cost = {best_vest_cost:.4f}")

    return visual_fit, vest_fit, best_vis_result, best_vest_result




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