import numpy as np
import scipy.optimize as opt
from Datasetcode import dataset

# ----------------------------
# Models (your code, unchanged)
# ----------------------------
def Hnm_model(w, omega_nm, zeta_nm):
    s = 1j * w
    return omega_nm**2 / (s**2 + 2*zeta_nm*omega_nm*s + omega_nm**2)

def Hpe_model(w, Kp, TL, TI, tau, omega_nm, zeta_nm):
    s = 1j * w
    equalization = (TL * s + 1) / (TI * s + 1)
    nm = Hnm_model(w, omega_nm, zeta_nm)
    delay = np.exp(-s * tau)
    return Kp * equalization * nm * delay

def Hsc_model(w, Tsc1, Tsc2, Tsc3):
    s = 1j * w
    return (1 + Tsc1 * s) / ((1 + Tsc2 * s) * (1 + Tsc3 * s))

def Hpxd_model(w, Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm, zeta_nm):
    s = 1j * w
    sc = Hsc_model(w, Tsc1, Tsc2, Tsc3)
    nm = Hnm_model(w, omega_nm, zeta_nm)
    delay = np.exp(-s * tau_m)
    return Km * s * sc * delay * nm

# ----------------------------
# Cost (vectorized, complex)
# ----------------------------
def cost_complex_relative(y_data, y_model, eps=1e-12):
    """
    Relative squared error on complex values, using real+imag parts.
    """
    denom = np.abs(y_data)**2 + eps
    err2 = (np.real(y_data - y_model)**2 + np.imag(y_data - y_model)**2)
    return np.sum(err2 / denom)

# ----------------------------
# fmin/minimize wrapper
# ----------------------------
def objective(theta, w, H_vis_data, H_vest_data, weight_vis=1.0, weight_vest=1.0):
    # unpack parameters
    (Kp, TL, TI, tau,
     Km, Tsc1, Tsc2, Tsc3, tau_m,
     omega_nm, zeta_nm) = theta

    # model predictions (arrays over w)
    H_vis_model  = Hpe_model(w, Kp, TL, TI, tau, omega_nm, zeta_nm)
    H_vest_model = Hpxd_model(w, Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm, zeta_nm)

    # costs
    J_vis  = cost_complex_relative(H_vis_data,  H_vis_model)
    J_vest = cost_complex_relative(H_vest_data, H_vest_model)

    return weight_vis * J_vis + weight_vest * J_vest

def fit_params(i,j):
    # ---- Load your dataset ----
    # Adjust these keys/fields to match Datasetcode.dataset structure.
    # Common patterns:
    # dataset["w"], dataset["H_vis"], dataset["H_vest"]
    w = dataset[i][j]["w_FC"]                      # rad/s (array)
    H_vis_data = dataset[i][j]["Hpe_FC"]         # complex FRF array
    H_vest_data = dataset[i][j]["Hpxd_FC"]       # complex FRF array

    # ---- Initial guess (use yours or something sensible) ----
    x0 = np.array([
        1.0,  # Kp
        1.0,  # TL
        1.0,  # TI
        0.1,  # tau (delay)
        1.0,  # Km
        1.0,  # Tsc1
        1.0,  # Tsc2
        1.0,  # Tsc3
        0.1,  # tau_m
        1.0,  # omega_nm
        0.7,  # zeta_nm
    ], dtype=float)

    # ---- Bounds: keep physically meaningful positive params ----
    # If Kp/Km can be negative in your identification, widen those.
    bounds = [
        (1e-6, None),  # Kp
        (1e-6, None),  # TL
        (1e-6, None),  # TI
        (0.0,  None),  # tau
        (1e-6, None),  # Km
        (1e-6, None),  # Tsc1
        (1e-6, None),  # Tsc2
        (1e-6, None),  # Tsc3
        (0.0,  None),  # tau_m
        (1e-6, None),  # omega_nm
        (1e-6, None),  # zeta_nm
    ]

    res = opt.minimize(
        objective,
        x0,
        args=(w, H_vis_data, H_vest_data, 1.0, 1.0),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000}
    )

    return res

if __name__ == "__main__":
    res = fit_params(1,1)
    print("Success:", res.success)
    print("Message:", res.message)
    print("Final cost:", res.fun)
    print("Theta*:", res.x)