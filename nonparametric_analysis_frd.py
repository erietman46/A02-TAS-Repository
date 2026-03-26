import numpy as np
from Datasetcode import dataset
import matplotlib.pyplot as plt
import control as ct


#DEFINE VEHICLE DYNAMICS: POSITION, VELOCITY, ACCELERATION

def controlled_element_frf(w, condition, Kc=1.0):
    s = 1j * w

    if condition == 1 or condition == 4:
        Hc = Kc * np.ones_like(s, dtype=complex)
    elif condition == 2 or condition == 5:
        Hc = Kc / s
    elif condition == 3 or condition == 6:
        Hc = Kc / (s ** 2)
    else:
        raise ValueError("vehicle_type must be 'position', 'velocity', or 'acceleration'")

    return Hc


#DEFINE OPEN LOOP TRANSFER FUNCTIONS FOR BOTH FIXED BASED AND MOTION BASED, SEPARATING TARGET AND DISTURBANCE

#Open loop transfer function for fixed base: L_t = L_d = Hpe * Hc

#Open loop transfer functions for motion based: L_t = (Hpe * Hc)/(1+sHpxd*Hc),    L_d = (Hpe + sHpxd)*Hc

def open_loop_transfer_function_motion(subject, condition, Kc=1.0):

    if condition not in [4, 5, 6]:
        raise ValueError("Motion-base conditions are [4, 5, 6].")

    Hpe = dataset[subject][condition]["Hpe_FC"]
    Hpxd = dataset[subject][condition]["Hpxd_FC"]
    w = dataset[subject][condition]["w_FC"]


    Hc = controlled_element_frf(w, condition, Kc)
    s = 1j*w

    H_ol_d = Hc * (Hpe + s* Hpxd)

    H_ol_t = (Hpe * Hc)/(1 + s * Hpxd*Hc)

    return w, H_ol_d, H_ol_t

def open_loop_transfer_function_fixed(subject, condition, Kc=1.0):

    Hpe = dataset[subject][condition]["Hpe"]
    w = dataset[subject][condition]["w"]
    
    Hc = controlled_element_frf(w, condition, Kc)
    s = 1j*w

    H_ol_t = Hc * Hpe

    return w, H_ol_t, H_ol_t # H_ol_t = H_ol_d


def open_loop_transfer_functions(subject, condition, Kc=1.0):
    if condition in [1, 2, 3]:
        return open_loop_transfer_function_fixed(subject, condition, Kc)
    elif condition in [4, 5, 6]:
        return open_loop_transfer_function_motion(subject, condition, Kc)
    else:
        raise ValueError("Condition must be 1..6")


#Helpers

def mag_db_phase(H):
    """Magnitude in dB and properly unwrapped phase in degrees."""
    mag = 20 * np.log10(np.abs(H))
    phase = np.unwrap(np.angle(H)) * 180 / np.pi
    return mag, phase


def condition_name(condition):
    names = {
        1: "Fixed-base Position",
        2: "Fixed-base Velocity",
        3: "Fixed-base Acceleration",
        4: "Motion-base Position",
        5: "Motion-base Velocity",
        6: "Motion-base Acceleration",
    }
    return names.get(condition, f"Condition {condition}")



#Plotting

def control_bode_and_margins(subject, condition, Kc=1.0, loop='target'):

    w, L_d, L_t = open_loop_transfer_functions(subject, condition, Kc)

    # Select loop
    if loop == 'target':
        L = L_t
    elif loop == 'disturbance':
        L = L_d
    else:
        raise ValueError("loop must be 'target' or 'disturbance'")

    # Convert to arrays
    w = np.asarray(w).flatten()
    L = np.asarray(L).flatten()

    # Remove NaN / Inf
    mask = np.isfinite(L) & np.isfinite(w)
    w = w[mask]
    L = L[mask]

    # Sort (CRITICAL)
    idx = np.argsort(w)
    w = w[idx]
    L = L[idx]

    # Create FRD system
    sys_frd = ct.frd(L.reshape(1, 1, -1), w, smooth = True, name=f"S{subject}_C{condition}_{loop}")


    # Bode plot
    ct.bode_plot(sys_frd, dB=True, deg=True)
    plt.suptitle(f"{condition_name(condition)} ({loop} loop)")
    plt.show()

    # Margins
    gm, pm, wg, wp = ct.margin(sys_frd)

    print("\n--- Stability Margins ---")
    print(f"Gain margin: {gm}")
    print(f"Phase margin: {pm}")
    print(f"Gain crossover frequency: {wg}")
    print(f"Phase crossover frequency: {wp}")

    return gm, pm, wg, wp


control_bode_and_margins(subject=1, condition=4, loop='disturbance')