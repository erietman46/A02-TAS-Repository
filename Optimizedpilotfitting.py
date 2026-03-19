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

def initialization(condition):
    """Return initial guess and bounds for a given condition."""
    motion = condition in [4, 5, 6]

    if motion:
        x0 = np.array([
            10,   # Kp
            1,   # TL
            2.5,   # TI
            0.25,   # tau
            15.0,  # omega_nm_vis
            0.5,   # zeta_nm_vis
            0,   # Km
            2.5,   # Tsc1
            2.5,   # Tsc2
            2.5,   # Tsc3
            0.25,   # tau_m
            15,  # omega_nm_vest
            0.5,   # zeta_nm_vest
        ])

        bounds = [
            (1e-6, None),  # Kp
            (1e-6, None),  # TL
            (1e-6, None),  # TI
            (0.0, None),   # tau
            (1e-6, None),  # omega_nm_vis
            (1e-6, None),  # zeta_nm_vis
            (1e-6, None),  # Km
            (1e-6, None),  # Tsc1
            (1e-6, None),  # Tsc2
            (1e-6, None),  # Tsc3
            (0.0, None),   # tau_m
            (1e-6, None),  # omega_nm_vest
            (1e-6, None),  # zeta_nm_vest
        ]
    else:
        x0 = np.array([
            10,   # Kp
            1,   # TL
            2.5,   # TI
            0.25,   # tau
            15.0,  # omega_nm_vis
            0.5,   # zeta_nm_vis
        ])

        bounds = [
            (1e-6, None),  # Kp
            (1e-6, None),  # TL
            (1e-6, None),  # TI
            (0.0, None),   # tau
            (1e-6, None),  # omega_nm_vis
            (1e-6, None),  # zeta_nm_vis
        ]

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




def fit_subject_condition(subject, condition, weight_vis=1.0, weight_vest=1.0):
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

    result = opt.minimize(
        cost_function,
        x0=x0,
        args=(w_FC, vis_data, vest_data, condition, weight_vis, weight_vest),
        method='L-BFGS-B',   # supports bounds
        # bounds=bounds,
    )

    fitted_params = result.x

    # Build fitted responses
    visual_fit = Hpe_model(w_FC, *fitted_params[:6])
    vest_fit = Hpxd_model(w_FC, *fitted_params[6:]) if motion else None

    # Store fitted parameters
    names = parameter_names(condition)
    dataset[subject][condition]['fitted_params'] = dict(zip(names, fitted_params))
    dataset[subject][condition]['fit_cost'] = result.fun
    dataset[subject][condition]['fit_success'] = result.success
    dataset[subject][condition]['fit_message'] = result.message

    return visual_fit, vest_fit, result, result.fun


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
    


