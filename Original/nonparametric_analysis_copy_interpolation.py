import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from Datasetcode import dataset


# =============================================================================
# CONTROLLED ELEMENT FRF
# Conditions:
#   1, 4 -> position vehicle       : Hc = Kc
#   2, 5 -> velocity vehicle       : Hc = Kc / s
#   3, 6 -> acceleration vehicle   : Hc = Kc / s^2
# =============================================================================

def controlled_element_frf(w, condition, Kc=1.0):
    w = np.asarray(w).ravel().astype(float)
    s = 1j * w

    if condition in [1, 4]:
        Hc = Kc * np.ones_like(s, dtype=complex)
    elif condition in [2, 5]:
        Hc = Kc / s
    elif condition in [3, 6]:
        Hc = Kc / (s ** 2)
    else:
        raise ValueError("Condition must be one of [1, 2, 3, 4, 5, 6].")

    return Hc


# =============================================================================
# OPEN-LOOP TRANSFER FUNCTIONS
# Fixed-base:
#   L_t = L_d = Hpe * Hc
#
# Motion-base:
#   L_t = (Hpe * Hc) / (1 + s*Hpxd*Hc)
#   L_d = (Hpe + s*Hpxd) * Hc
# =============================================================================

def open_loop_transfer_function_motion(subject, condition, Kc=1.0):
    if condition not in [4, 5, 6]:
        raise ValueError("Motion-base conditions are [4, 5, 6].")

    rec = dataset[subject][condition]

    # Correct dataset keys
    Hpe = np.asarray(rec["Hpe_FC"]).ravel()
    Hpxd = np.asarray(rec["Hpxd_FC"]).ravel()
    w = np.asarray(rec["w_FC"]).ravel().astype(float)

    Hc = controlled_element_frf(w, condition, Kc)
    s = 1j * w

    H_ol_d = (Hpe + s * Hpxd) * Hc
    H_ol_t = (Hpe * Hc) / (1 + s * Hpxd * Hc)

    return w, H_ol_d, H_ol_t


def open_loop_transfer_function_fixed(subject, condition, Kc=1.0):
    if condition not in [1, 2, 3]:
        raise ValueError("Fixed-base conditions are [1, 2, 3].")

    rec = dataset[subject][condition]

    # Correct dataset keys
    Hpe = np.asarray(rec["Hpe_FC"]).ravel()
    w = np.asarray(rec["w_FC"]).ravel().astype(float)

    Hc = controlled_element_frf(w, condition, Kc)

    H_ol_t = Hc * Hpe
    H_ol_d = H_ol_t.copy()

    return w, H_ol_d, H_ol_t


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


# =============================================================================
# HELPERS
# =============================================================================

def magnitude_db(H):
    return 20 * np.log10(np.maximum(np.abs(H), 1e-12))


def phase_deg(H):
    # unwrap in radians, then convert
    return np.unwrap(np.angle(H)) * 180 / np.pi


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


def crossover_and_phase_margin(w, L):
    """
    Estimate crossover frequency wc and phase margin PM from nonparametric FRF.

    wc: first frequency where |L| crosses 1
    PM: 180 + phase(L(j wc))   [deg]
    """
    w = np.asarray(w).ravel().astype(float)
    L = np.asarray(L).ravel()

    mag = np.abs(L)
    ph = phase_deg(L)

    # Find first sign change in (mag - 1)
    idx = np.where(np.diff(np.sign(mag - 1.0)) != 0)[0]

    if len(idx) == 0:
        return np.nan, np.nan

    i = idx[0]

    # interpolate crossover in log-frequency space
    x1, x2 = np.log10(w[i]), np.log10(w[i + 1])
    y1, y2 = mag[i], mag[i + 1]

    if np.isclose(y2, y1):
        wc = w[i]
    else:
        x_cross = x1 + (1.0 - y1) * (x2 - x1) / (y2 - y1)
        wc = 10 ** x_cross

    # interpolate phase at wc
    p1, p2 = ph[i], ph[i + 1]
    if np.isclose(x2, x1):
        ph_wc = p1
    else:
        ph_wc = p1 + (np.log10(wc) - x1) * (p2 - p1) / (x2 - x1)

    pm = 180.0 + ph_wc
    return wc, pm




#INTERPOLATION
def rbf_bode_interpolation(w, H, n_points=400, smooth=0.0, function='multiquadric'):
    """
    Radial basis interpolation of FRF magnitude and phase over log-frequency.

    Parameters
    ----------
    w : array_like
        Frequency vector [rad/s]
    H : array_like
        Complex FRF values
    n_points : int
        Number of interpolated frequency points
    smooth : float
        RBF smoothing parameter
    function : str
        RBF kernel, e.g. 'multiquadric', 'linear', 'cubic', 'quintic', 'thin_plate'

    Returns
    -------
    w_fine : ndarray
        Dense frequency grid
    mag_fine : ndarray
        Interpolated magnitude [dB]
    ph_fine : ndarray
        Interpolated phase [deg]
    """
    w = np.asarray(w).ravel().astype(float)
    H = np.asarray(H).ravel()

    x = np.log10(w)
    mag = magnitude_db(H)
    ph = phase_deg(H)

    x_fine = np.linspace(x.min(), x.max(), n_points)
    w_fine = 10 ** x_fine

    rbf_mag = Rbf(x, mag, function=function, smooth=smooth)
    rbf_ph = Rbf(x, ph, function=function, smooth=smooth)

    mag_fine = rbf_mag(x_fine)
    ph_fine = rbf_ph(x_fine)

    return w_fine, mag_fine, ph_fine


def crossover_and_phase_margin_from_bode(w, mag_db_vals, ph_deg_vals):
    """
    Estimate crossover frequency and phase margin from interpolated Bode data.

    Parameters
    ----------
    w : ndarray
        Frequency vector
    mag_db_vals : ndarray
        Magnitude in dB
    ph_deg_vals : ndarray
        Phase in deg

    Returns
    -------
    wc : float
        Crossover frequency [rad/s]
    pm : float
        Phase margin [deg]
    """
    w = np.asarray(w).ravel().astype(float)
    mag_db_vals = np.asarray(mag_db_vals).ravel()
    ph_deg_vals = np.asarray(ph_deg_vals).ravel()

    idx = np.where(np.diff(np.sign(mag_db_vals)) != 0)[0]

    if len(idx) == 0:
        return np.nan, np.nan

    i = idx[0]

    x1, x2 = np.log10(w[i]), np.log10(w[i + 1])
    y1, y2 = mag_db_vals[i], mag_db_vals[i + 1]

    if np.isclose(y2, y1):
        wc = w[i]
    else:
        x_cross = x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)
        wc = 10 ** x_cross

    p1, p2 = ph_deg_vals[i], ph_deg_vals[i + 1]
    if np.isclose(x2, x1):
        ph_wc = p1
    else:
        ph_wc = p1 + (np.log10(wc) - x1) * (p2 - p1) / (x2 - x1)

    pm = 180.0 + ph_wc
    return wc, pm



# =============================================================================
# PLOTTING
# =============================================================================

def plot_open_loops(subject, condition, Kc=1.0, show=True, savepath=None,
                    smooth_curves=True, n_points=400, rbf_smooth=0.0, rbf_function='multiquadric'):
    """
    Plot target and disturbance Bode plots for one subject and condition.
    Uses RBF interpolation for smooth curves.
    """
    w, L_d, L_t = open_loop_transfer_functions(subject, condition, Kc)

    mag_d = magnitude_db(L_d)
    mag_t = magnitude_db(L_t)
    ph_d = phase_deg(L_d)
    ph_t = phase_deg(L_t)

    if smooth_curves:
        w_d_fine, mag_d_fine, ph_d_fine = rbf_bode_interpolation(
            w, L_d, n_points=n_points, smooth=rbf_smooth, function=rbf_function
        )
        w_t_fine, mag_t_fine, ph_t_fine = rbf_bode_interpolation(
            w, L_t, n_points=n_points, smooth=rbf_smooth, function=rbf_function
        )

        wc_d, pm_d = crossover_and_phase_margin_from_bode(w_d_fine, mag_d_fine, ph_d_fine)
        wc_t, pm_t = crossover_and_phase_margin_from_bode(w_t_fine, mag_t_fine, ph_t_fine)
    else:
        w_d_fine, mag_d_fine, ph_d_fine = w, mag_d, ph_d
        w_t_fine, mag_t_fine, ph_t_fine = w, mag_t, ph_t
        wc_d, pm_d = crossover_and_phase_margin(w, L_d)
        wc_t, pm_t = crossover_and_phase_margin(w, L_t)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Magnitude
    axes[0].semilogx(w_d_fine, mag_d_fine, '-', linewidth=2,
                     label=f'Disturbance $L_d$ | wc={wc_d:.3f}, PM={pm_d:.1f}°')
    axes[0].semilogx(w_t_fine, mag_t_fine, '-', linewidth=2,
                     label=f'Target $L_t$ | wc={wc_t:.3f}, PM={pm_t:.1f}°')

    # original points
    axes[0].semilogx(w, mag_d, 'o', markersize=4)
    axes[0].semilogx(w, mag_t, 's', markersize=4)

    axes[0].axhline(0, color='k', linestyle='--', linewidth=1)
    if np.isfinite(wc_d):
        axes[0].axvline(wc_d, linestyle=':', linewidth=1)
    if np.isfinite(wc_t):
        axes[0].axvline(wc_t, linestyle=':', linewidth=1)

    axes[0].set_ylabel('Magnitude [dB]')
    axes[0].set_title(f'Subject {subject} - {condition_name(condition)}')
    axes[0].grid(True, which='both', linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=9)

    # Phase
    axes[1].semilogx(w_d_fine, ph_d_fine, '-', linewidth=2,
                     label=f'Disturbance $L_d$ | PM={pm_d:.1f}°')
    axes[1].semilogx(w_t_fine, ph_t_fine, '-', linewidth=2,
                     label=f'Target $L_t$ | PM={pm_t:.1f}°')

    # original points
    axes[1].semilogx(w, ph_d, 'o', markersize=4)
    axes[1].semilogx(w, ph_t, 's', markersize=4)

    axes[1].axhline(-180, color='k', linestyle='--', linewidth=1)
    if np.isfinite(wc_d):
        axes[1].axvline(wc_d, linestyle=':', linewidth=1)
    if np.isfinite(wc_t):
        axes[1].axvline(wc_t, linestyle=':', linewidth=1)

    axes[1].set_xlabel('Frequency [rad/s]')
    axes[1].set_ylabel('Phase [deg]')
    axes[1].grid(True, which='both', linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=9)

    metrics_text = (
        f"Disturbance: wc = {wc_d:.3f} rad/s, PM = {pm_d:.1f}°\n"
        f"Target:      wc = {wc_t:.3f} rad/s, PM = {pm_t:.1f}°"
    )
    axes[1].text(
        0.02, 0.03, metrics_text,
        transform=axes[1].transAxes,
        fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "subject": subject,
        "condition": condition,
        "w": w,
        "L_d": L_d,
        "L_t": L_t,
        "wc_d": wc_d,
        "pm_d": pm_d,
        "wc_t": wc_t,
        "pm_t": pm_t,
    }


def plot_all_subjects_all_conditions(Kc=1.0, show=True, save_dir=None):
    """
    Plot all available subject/condition combinations.
    Returns a nested dict of metrics.
    """
    results = {}

    for subject in sorted(dataset.keys()):
        results[subject] = {}

        for condition in sorted(dataset[subject].keys()):
            if condition not in [1, 2, 3, 4, 5, 6]:
                continue

            savepath = None
            if save_dir is not None:
                import os
                os.makedirs(save_dir, exist_ok=True)
                savepath = os.path.join(
                    save_dir,
                    f"subject_{subject}_condition_{condition}.png"
                )

            res = plot_open_loops(
                subject=subject,
                condition=condition,
                Kc=Kc,
                show=show,
                savepath=savepath
            )

            results[subject][condition] = {
                "wc_d": res["wc_d"],
                "pm_d": res["pm_d"],
                "wc_t": res["wc_t"],
                "pm_t": res["pm_t"],
            }

    return results


def print_all_metrics(Kc=1.0):
    for subject in sorted(dataset.keys()):
        print(f"\n=== Subject {subject} ===")
        for condition in sorted(dataset[subject].keys()):
            if condition not in [1, 2, 3, 4, 5, 6]:
                continue

            w, L_d, L_t = open_loop_transfer_functions(subject, condition, Kc)
            wc_d, pm_d = crossover_and_phase_margin(w, L_d)
            wc_t, pm_t = crossover_and_phase_margin(w, L_t)

            print(
                f"Condition {condition} ({condition_name(condition)}): "
                f"L_d -> wc={wc_d:.3f}, PM={pm_d:.1f}° | "
                f"L_t -> wc={wc_t:.3f}, PM={pm_t:.1f}°"
            )






# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Loop through every subject and every condition
    for subject in sorted(dataset.keys()):
        for condition in sorted(dataset[subject].keys()):
            if condition not in [1, 2, 3, 4, 5, 6]:
                continue

            print(f"\n--- Subject {subject}, Condition {condition} ({condition_name(condition)}) ---")

            # Plot
            plot_open_loops(subject=subject, condition=condition, Kc=1.0, show=True)

            # Metrics
            w, L_d, L_t = open_loop_transfer_functions(subject, condition, Kc=1.0)
            wc_d, pm_d = crossover_and_phase_margin(w, L_d)
            wc_t, pm_t = crossover_and_phase_margin(w, L_t)

            print(f"Disturbance loop: wc = {wc_d:.3f} rad/s, PM = {pm_d:.1f} deg")
            print(f"Target loop     : wc = {wc_t:.3f} rad/s, PM = {pm_t:.1f} deg")


if __name__ == "__main__":
    for subject in sorted(dataset.keys()):
        for condition in sorted(dataset[subject].keys()):
            if condition not in [1, 2, 3, 4, 5, 6]:
                continue

            print(f"\n--- Subject {subject}, Condition {condition} ({condition_name(condition)}) ---")

            plot_open_loops(
                subject=subject,
                condition=condition,
                Kc=1.0,
                show=True,
                smooth_curves=True,
                n_points=500,
                rbf_smooth=0.05,
                rbf_function='multiquadric'
            )