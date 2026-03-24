#optimisedpilotfitting_withAI.py
import numpy as np
import scipy.optimize as opt
from scipy.stats import qmc
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


def Hpxd_model(w, omega_nm, zeta_nm, Km, Tsc1, Tsc2, Tsc3, tau_m):
    """
    Vestibular pilot model (Hpxd).
    Params: Km, Tsc1, Tsc2, Tsc3, tau_m,
    """
    s = 1j * w
    sc = Hsc_model(w, Tsc1, Tsc2, Tsc3)
    nm = Hnm_model(w, omega_nm, zeta_nm)
    delay = np.exp(-s * tau_m)
    return Km * s * sc * delay * nm


# ==============================================================================
# COST FUNCTION
# ==============================================================================

def make_cost_fn(w, vis_data, vest_data, condition, weight_vis=1.0, weight_vest=1.0):
    """
    Return a cost function with pre-computed denominators.

    Pre-computing |data|^2 once avoids repeating this calculation on every
    optimizer iteration (~N_STARTS * maxiter times per fit).

    Parameter vector layout depends on condition:
      - No-motion (C1-C3): only visual params are meaningful.
        params = [Kp, TL, TI, tau, omega_nm, zeta_nm]
      - Motion (C4-C6): visual + vestibular params fitted together.
        params = [Kp, TL, TI, tau, omega_nm, zeta_nm,
                  Km, Tsc1, Tsc2, Tsc3, tau_m]
    """
    vis_denom = np.abs(vis_data)**2 + 1e-12
    vest_denom = (np.abs(vest_data)**2 + 1e-12) if vest_data is not None else None
    motion = condition in [4, 5, 6]

    def _cost(params):
        Kp, TL, TI, tau, omega_nm, zeta_nm = params[:6]
        vis_model = Hpe_model(w, Kp, TL, TI, tau, omega_nm, zeta_nm)
        cost = weight_vis * np.sum(np.abs(vis_data - vis_model)**2 / vis_denom)
        if motion:
            omega_nm_v, zeta_nm_v, Km, Tsc1, Tsc2, Tsc3, tau_m = params[4:]
            vest_model = Hpxd_model(w, omega_nm_v, zeta_nm_v, Km, Tsc1, Tsc2, Tsc3, tau_m)
            cost += weight_vest * np.sum(np.abs(vest_data - vest_model)**2 / vest_denom)
        return float(np.real(cost))

    return _cost


# ==============================================================================
# PARAMETER BOUNDS & INITIAL GUESSES
# ==============================================================================

VIS_BOUNDS = [
    (0.01, 10.0),   # Kp
    (0.0,  10.0),   # TL
    (0.0,  10.0),   # TI
    (0.01, 0.5),    # tau  [s]
    (5.0,  35.0),   # omega_nm  [rad/s]
    (0.1,  1.0),    # zeta_nm
]

# Vestibular (Hpxd) parameter bounds: Km, Tsc1, Tsc2, Tsc3, tau_m
VEST_BOUNDS = [
    (-5.0,  5.0),    # Km  (can be negative — motion opposition)
    (0.11,  0.11),   # Tsc1
    (5.924, 5.924),  # Tsc2
    (0.005, 0.005),  # Tsc3
    (0.01,  1.5),    # tau_m  [s]
]

def _mid(bounds):
    return np.array([(lo + hi) / 2 for lo, hi in bounds], dtype=float)


VIS_X0 = _mid(VIS_BOUNDS)
VEST_X0 = _mid(VEST_BOUNDS)

# Latin Hypercube gives better parameter space coverage than random uniform,
# so fewer starts are needed for equivalent quality.
N_STARTS = 15


def lhs_starts(bounds, n):
    """Generate n starts: midpoint + (n-1) Latin Hypercube Sampling points within bounds.
    Fixed parameters (lo == hi) are excluded from LHS and held at their fixed value."""
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    midpoint = (lo + hi) / 2
    free_idx = np.where(lo < hi)[0]
    sampler = qmc.LatinHypercube(d=len(free_idx), seed=11)
    unit = sampler.random(n - 1)
    scaled = qmc.scale(unit, lo[free_idx], hi[free_idx])
    starts = [midpoint]
    for k in range(n - 1):
        x = midpoint.copy()
        x[free_idx] = scaled[k]
        starts.append(x)
    return starts


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
        base_names += ['Km', 'Tsc1', 'Tsc2', 'Tsc3', 'tau_m']
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

    # Build cost function once — denominators are pre-computed here, not per iteration
    cost_fn = make_cost_fn(w, vis_data, vest_data, condition, weight_vis, weight_vest)

    best_cost = np.inf
    best_params = None
    best_result = None

    starts = lhs_starts(bounds, N_STARTS)

    for x0 in starts:
        try:
            res = opt.minimize(
                cost_fn,
                x0=x0,
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
    vest_fit = Hpxd_model(w, *fitted_params[4:]) if motion else None

    names = parameter_names(condition)
    dataset[subject][condition]["fitted_params"] = dict(zip(names, fitted_params))
    dataset[subject][condition]["fit_cost"] = best_cost
    dataset[subject][condition]["fit_success"] = best_result.success
    dataset[subject][condition]["fit_message"] = best_result.message

    if verbose:
        print(f"  Subj {subject}, Cond {condition}  →  cost = {best_cost:.4f}")

    return visual_fit, vest_fit, best_result, best_cost