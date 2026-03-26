import numpy as np
import Datasetcode as ds


#DEFINE VEHICLE DYNAMICS: POSITION, VELOCITY, ACCELERATION

def controlled_element_frf(w, vehicle_type="position", Kc=1.0):
    s = 1j * w

    if vehicle_type.lower() == "position":
        Hc = Kc * np.ones_like(s, dtype=complex)
    elif vehicle_type.lower() == "velocity":
        Hc = Kc / s
    elif vehicle_type.lower() == "acceleration":
        Hc = Kc / (s ** 2)
    else:
        raise ValueError("vehicle_type must be 'position', 'velocity', or 'acceleration'")

    return Hc


#DEFINE OPEN LOOP TRANSFER FUNCTIONS FOR BOTH FIXED BASED AND MOTION BASED 

def open_loop_transfer_functions(w, Hpe, Hpxd, vehicle_type="position", Kc=1.0, sign_motion=+1):
    """
    Compute visual, motion, and total open-loop FRFs.

    Parameters
    ----------
    w : array_like
        Frequency vector [rad/s]
    Hpe : array_like of complex
        Visual pilot FRF
    Hpxd : array_like of complex
        Vestibular pilot FRF (from x_dot to u)
    vehicle_type : str
        'position', 'velocity', or 'acceleration'
    Kc : float
        Vehicle gain
    sign_motion : int
        +1 or -1, depending on sign convention

    Returns
    -------
    dict with:
        Hc, L_visual, L_motion, L_total
    """
    w = np.asarray(w).flatten()
    Hpe = np.asarray(Hpe).flatten()
    Hpxd = np.asarray(Hpxd).flatten()
    
    s = 1j * w
    Hc = controlled_element_frf(w, vehicle_type=vehicle_type, Kc=Kc)

    L_visual = Hpe * Hc
    L_motion = (s * Hpxd) * Hc
    L_total = (Hpe + sign_motion * s * Hpxd) * Hc

    return {
        "Hc": Hc,
        "L_visual": L_visual,
        "L_motion": L_motion,
        "L_total": L_total,
    }

