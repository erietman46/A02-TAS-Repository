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

