import numpy as np
from Datasetcode import dataset
import matplotlib.pyplot as plt

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

    Hpe = dataset[subject][condition]["Hpe"]
    Hpxd = dataset[subject][condition]["Hpxd"]
    w = dataset[subject][condition]["w"]

    

    Hc = controlled_element_frf(w, condition, Kc)
    s = 1j*w

    H_ol_d = Hc * (Hpe + s* Hpxd)

    H_ol_t = (Hpe * Hc)/(1 + s * Hpxd*Hc)

    return H_ol_d, H_ol_t

def open_loop_transfer_function_fixed(subject, condition, Kc=1.0):

    Hpe = dataset[subject][condition]["Hpe"]
    w = dataset[subject][condition]["w"]
    
    Hc = controlled_element_frf(w, condition, Kc)
    s = 1j*w

    H_ol_t = Hc * Hpe

    return H_ol_t

#Merge the two functions into one that can handle both fixed and motion based conditions, returning both L_d and L_t (with L_d = L_t for fixed based conditions)
def open_loop_transfer_functions(subject, condition, Kc=1.0):
    """
    Returns:
        w, L_d, L_t
    for any condition 1..6
    """
    if condition in [1, 2, 3]:
        return open_loop_transfer_function_fixed(subject, condition, Kc)
    elif condition in [4, 5, 6]:
        return open_loop_transfer_function_motion(subject, condition, Kc)
    else:
        raise ValueError("Condition must be one of [1, 2, 3, 4, 5, 6].")


#Helpers

def mag_db(H):
    """Return magnitude in dB and unwrapped phase in degrees."""
    H_abs = np.abs(H)
    H_db = 20 * np.log10(H_abs)
    H_ang = np.angle(H, deg=True)
    H_ang = np.unwrap(H_ang, period=360, axis=0)
    return H_db, H_ang


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

def plot_open_loops(subject, condition, Kc=1.0, show=True):
    """
    Plot disturbance and target open-loop FRFs for one subject and condition.
    """
    w, L_d, L_t = open_loop_transfer_functions(subject, condition, Kc)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Magnitude
    axes[0].semilogx(w, mag_db(L_d)[0], 'o-', label='Disturbance open loop $L_d$')
    axes[0].semilogx(w, mag_db(L_t)[0], 's-', label='Target open loop $L_t$')
    axes[0].set_ylabel('Magnitude [dB]')
    axes[0].set_title(f'Subject {subject} - {condition_name(condition)}')
    axes[0].grid(True, which='both', linestyle='--', alpha=0.6)
    axes[0].legend()

    # Phase
    axes[1].semilogx(w, mag_db(L_d)[1], 'o-', label='Disturbance open loop $L_d$')
    axes[1].semilogx(w, mag_db(L_t)[1], 's-', label='Target open loop $L_t$')
    axes[1].set_xlabel('Frequency [rad/s]')
    axes[1].set_ylabel('Phase [deg]')
    axes[1].grid(True, which='both', linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes

plot_open_loops(subject=1, condition=4)




