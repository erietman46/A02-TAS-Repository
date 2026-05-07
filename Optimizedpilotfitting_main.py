#optimisedpilotfitting_main.py
import numpy as np
import scipy.optimize as opt
from Datasetcode import dataset

# Model definitions
def Hnm_model(w, omega_nm, zeta_nm):
    s = 1j * w
    return omega_nm**2 / (s**2 + 2 * zeta_nm * omega_nm * s + omega_nm**2)

def Hpe_model(w, Kp, TL, TI, tau, omega_nm, zeta_nm):
    s = 1j * w
    equalization = (TL * s + 1) / (TI * s + 1)
    nm = Hnm_model(w, omega_nm, zeta_nm)
    delay = np.exp(-s * tau)
    return Kp * equalization * nm * delay

def Hsc_model(w, Tsc1, Tsc2, Tsc3):
    s = 1j * w
    return (1 + Tsc1 * s) / ((1 + Tsc2 * s) * (1 + Tsc3 * s))

def Hpxd_model(w, omega_nm, zeta_nm, Km, Tsc1, Tsc2, Tsc3, tau_m):
    s = 1j * w
    sc = Hsc_model(w, Tsc1, Tsc2, Tsc3)
    nm = Hnm_model(w, omega_nm, zeta_nm)
    delay = np.exp(-s * tau_m)
    return Km * s * sc * delay * nm

#cost function definition for optimisation
def cost_function(params, w, vis_data, vest_data, condition, weight_vis=1.0, weight_vest=1.0):
    
    # Visual model
    Kp, TL, TI, tau, omega_nm, zeta_nm = params[:6]
    vis_model = Hpe_model(w, Kp, TL, TI, tau, omega_nm, zeta_nm)
    err_vis = np.abs(vis_data - vis_model)**2 / (np.abs(vis_data)**2 + 1e-12)
    cost = weight_vis * np.sum(err_vis)

    # Vestibular model
    if condition in [4, 5, 6]:
        omega_nm, zeta_nm, Km, Tsc1, Tsc2, Tsc3, tau_m = params[4:]
        vest_model = Hpxd_model(w, omega_nm, zeta_nm, Km, Tsc1, Tsc2, Tsc3, tau_m )
        err_vest = np.abs(vest_data - vest_model)**2 / (np.abs(vest_data)**2 + 1e-12)
        cost += weight_vest * np.sum(err_vest)

    return float(np.real(cost))

# Bounds initialisation
VIS_BOUNDS = [
    (0.01, 10.0),   # Kp
    (0.0,  10.5),    # TL
    (0.0, 10.5),    # TI
    (0.01, 0.5),    # tau  [s]
    (5.0,  35.0),   # omega_nm  [rad/s]
    (0.1,  1.0),    # zeta_nm
]
VEST_BOUNDS = [
    (-5.0,  5.0),   # Km 
    (0.11,   0.11),   # Tsc1
    (5.924,  5.924),   # Tsc2
    (0.005,  0.005),   # Tsc3
    (0.01,  1.5),   # tau_m  [s]
]

#get mid point of bounds for initial guess
def _mid(bounds):
    return np.array([(lo + hi) / 2 for lo, hi in bounds], dtype=float)

#get mid point for visual and vestibular
VIS_X0 = _mid(VIS_BOUNDS)
VEST_X0 = _mid(VEST_BOUNDS)

N_STARTS = 30  
RNG = np.random.default_rng(11) # use seed 42 for reproducibility

# get random location in bounded domain
def random_x0(bounds):
    return np.array([RNG.uniform(lo, hi) for lo, hi in bounds], dtype=float)


def initialization(condition):
    motion = condition in [4, 5, 6]

    if motion:
        x0 = np.concatenate([VIS_X0, VEST_X0])
        bounds = VIS_BOUNDS + VEST_BOUNDS
    else:
        x0 = VIS_X0.copy()
        bounds = VIS_BOUNDS.copy()

    return x0, bounds


def parameter_names(condition):
    base_names = ['Kp', 'TL', 'TI', 'tau', 'omega_nm', 'zeta_nm']
    #if motion add additional
    if condition in [4, 5, 6]:
        base_names += ['Km', 'Tsc1', 'Tsc2', 'Tsc3', 'tau_m' ]
    return base_names


def fit_subject_condition(subject, condition, weight_vis=1.0, weight_vest=1.0, verbose=True):
   
    motion = condition in [4, 5, 6]

    rec = dataset[subject][condition]

    w = np.asarray(rec["w_FC"]).ravel().astype(float)
    vis_data = np.asarray(rec["Hpe_FC"]).ravel()
    vest_data = np.asarray(rec["Hpxd_FC"]).ravel() if motion else None

    x0_base, bounds = initialization(condition)
    #in the beginning set best cost to infinity and best params to None
    best_cost = np.inf
    best_params = None
    best_result = None

    starts = [x0_base] + [random_x0(bounds) for _ in range(N_STARTS - 1)]

    for x0 in starts:
        
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

visual_fit, vest_fit, result, result.fun = fit_subject_condition(1,4)
