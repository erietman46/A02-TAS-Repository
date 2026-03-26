import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# pip install control
import control as ct

from Datasetcode import dataset


# =============================================================================
# Settings
# =============================================================================
OUTPUT_DIR = Path("open_loop_results")
OUTPUT_DIR.mkdir(exist_ok=True)

SAVE_FIGURES = True
SHOW_FIGURES = False
FIG_DPI = 220

POLY_DEGREE = 4
RIDGE_ALPHA = 1e-3
SMOOTH_POINTS = 500

USE_MEASURED_FC_DATA = True   # uses w_FC, Hpe_FC, Hpxd_FC when available
Kc = 1.0


# =============================================================================
# Plot style
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 14,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# =============================================================================
# Labels / naming
# =============================================================================
def condition_name(condition: int) -> str:
    names = {
        1: "Fixed-base Position",
        2: "Fixed-base Velocity",
        3: "Fixed-base Acceleration",
        4: "Motion-base Position",
        5: "Motion-base Velocity",
        6: "Motion-base Acceleration",
    }
    return names.get(condition, f"Condition {condition}")


def is_motion_condition(condition: int) -> bool:
    return condition in [4, 5, 6]


def is_valid_condition(condition: int) -> bool:
    return condition in [1, 2, 3, 4, 5, 6]


# =============================================================================
# Controlled element
# =============================================================================
def controlled_element_frf(w: np.ndarray, condition: int, kc: float = 1.0) -> np.ndarray:
    """
    Controlled element dynamics (realistic vehicle dynamics):

        condition 1,4 -> position
            Hc = Kc * 1000 / (s + 10)^3

        condition 2,5 -> velocity
            Hc = Kc * 3600 / (s * (s + 30)^2)

        condition 3,6 -> acceleration
            Hc = Kc * 15 / s^2
    """
    s = 1j * w

    if condition in [1, 4]:
        return kc * 1000 / (s + 10)**3

    if condition in [2, 5]:
        return kc * 3600 / (s * (s + 30)**2)

    if condition in [3, 6]:
        return kc * 15 / (s**2)

    raise ValueError("Condition must be one of [1, 2, 3, 4, 5, 6].")


# =============================================================================
# Dataset access
# =============================================================================
def get_measured_frequency_data(subject: int, condition: int):
    """
    Read measured pilot frequency-response data from Datasetcode.py.

    Expected keys from the dataset file:
        w_FC      : frequency vector
        Hpe_FC    : measured visual pilot FRF
        Hpxd_FC   : measured motion/vestibular pilot FRF (motion-base only)

    Falls back to w, Hpe, Hpxd if FC versions are not present.
    """
    rec = dataset[subject][condition]

    if USE_MEASURED_FC_DATA:
        w = np.asarray(rec.get("w_FC", rec.get("w"))).ravel().astype(float)
        hpe = np.asarray(rec.get("Hpe_FC", rec.get("Hpe"))).ravel().astype(complex)
        hpxd_raw = rec.get("Hpxd_FC", rec.get("Hpxd"))
    else:
        w = np.asarray(rec["w"]).ravel().astype(float)
        hpe = np.asarray(rec["Hpe"]).ravel().astype(complex)
        hpxd_raw = rec.get("Hpxd", None)

    hpxd = None if hpxd_raw is None else np.asarray(hpxd_raw).ravel().astype(complex)

    return w, hpe, hpxd


# =============================================================================
# Open-loop transfer functions
# =============================================================================
def compute_open_loop_transfer_functions(subject: int, condition: int, kc: float = 1.0):
    """
    Build disturbance and target open-loop transfer functions from measured data.

    Fixed-base:
        L_d = L_t = Hc * Hpe

    Motion-base:
        L_d = Hc * (Hpe + s * Hpxd)
        L_t = (Hpe * Hc) / (1 + s * Hpxd * Hc)

    Notes:
    - This follows the equations you wrote in your own code/comments.
    - It assumes measured Hpxd_FC corresponds to Hpxd, so the extra factor s is applied here.
    """
    if not is_valid_condition(condition):
        raise ValueError(f"Invalid condition: {condition}")

    w, hpe, hpxd = get_measured_frequency_data(subject, condition)
    h_c = controlled_element_frf(w, condition, kc=kc)
    s = 1j * w

    if not is_motion_condition(condition):
        l_t = h_c * hpe
        l_d = l_t.copy()
    else:
        if hpxd is None:
            raise ValueError(f"Subject {subject}, condition {condition} has no Hpxd data.")

        l_d = h_c * (hpe + s * hpxd)
        l_t = (hpe * h_c) / (1.0 + s * hpxd * h_c)

    return w, l_d, l_t


# =============================================================================
# Bode helpers
# =============================================================================
def magnitude_and_phase(H: np.ndarray):
    """
    Returns:
        magnitude (absolute, not dB)
        phase in degrees, unwrapped
    """
    mag = np.abs(H)
    phase_deg = np.unwrap(np.angle(H)) * 180.0 / np.pi
    return mag, phase_deg


def smooth_curve_with_polynomial_regression(
    w: np.ndarray,
    y: np.ndarray,
    degree: int = POLY_DEGREE,
    alpha: float = RIDGE_ALPHA,
    n_points: int = SMOOTH_POINTS,
):
    """
    Smooth a bode curve by fitting y(log10(w)) with polynomial regression.
    """
    x = np.log10(np.asarray(w).ravel())[:, None]
    y = np.asarray(y).ravel()

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        Ridge(alpha=alpha),
    )
    model.fit(x, y)

    x_dense = np.linspace(x.min(), x.max(), n_points)[:, None]
    y_dense = model.predict(x_dense)
    w_dense = 10 ** x_dense.ravel()

    return w_dense, y_dense


# =============================================================================
# Margin extraction with control
# =============================================================================
def compute_margins(w: np.ndarray, L: np.ndarray):
    """
    Compute classical margins from measured complex FRF samples.

    control.margin expects:
        magnitude (absolute, not dB),
        phase in degrees,
        omega
    """
    mag, phase_deg = magnitude_and_phase(L)

    # Wrap phase into [-180, 180) for control.margin
    phase_wrapped = ((phase_deg + 180.0) % 360.0) - 180.0

    gain_margin = np.nan
    phase_margin = np.nan
    phase_crossover = np.nan
    gain_crossover = np.nan

    try:
        gm, pm, wpc, wgc = ct.margin(mag, phase_wrapped, w)
        gain_margin = float(gm) if np.isfinite(gm) else np.nan
        phase_margin = float(pm) if np.isfinite(pm) else np.nan
        phase_crossover = float(wpc) if np.isfinite(wpc) else np.nan
        gain_crossover = float(wgc) if np.isfinite(wgc) else np.nan
    except Exception:
        pass

    return {
        "gain_margin": gain_margin,
        "phase_margin_deg": phase_margin,
        "phase_crossover_rad_s": phase_crossover,
        "gain_crossover_rad_s": gain_crossover,
    }


# =============================================================================
# Plot formatting
# =============================================================================
def setup_bode_axis(ax, ylabel: str, phase_plot: bool = False):
    ax.set_xscale("log")
    ax.minorticks_on()

    ax.grid(True, which="major", color="0.70")
    ax.grid(True, which="minor", color="0.70", linestyle=(0, (1.2, 4.5)))

    ax.set_xlabel(r"$\omega,\ \mathrm{rad\ s^{-1}}$")
    ax.set_ylabel(ylabel)
    ax.set_box_aspect(1)

    if phase_plot:
        ax.axhline(-180, color="0.4", linewidth=0.8)
    else:
        ax.axhline(1.0, color="0.4", linewidth=0.8)
        ax.set_yscale("log")


def draw_margin_annotation(ax, margins: dict, mode: str):
    """
    mode:
        "target" or "disturbance"
    """
    wc = margins["gain_crossover_rad_s"]
    pm = margins["phase_margin_deg"]

    if np.isfinite(wc):
        ax.axvline(wc, color="0.45", linewidth=0.8)

    if np.isfinite(pm) and np.isfinite(wc):
        y0 = -180
        y1 = -180 + pm
        ax.annotate(
            "",
            xy=(wc, y0),
            xytext=(wc, y1),
            arrowprops=dict(arrowstyle="-|>", color="0.45", lw=0.8),
        )

        if mode == "disturbance":
            text = rf"$\varphi_{{m,d}} = {pm:.1f}^\circ$"
        else:
            text = rf"$\varphi_{{m,t}} = {pm:.1f}^\circ$"

        ax.text(0.33, 0.27, text, transform=ax.transAxes, fontsize=12)


# =============================================================================
# Plot one subject-condition
# =============================================================================
def plot_subject_condition(subject: int, condition: int, kc: float = 1.0, outdir: Path = OUTPUT_DIR):
    w, L_d, L_t = compute_open_loop_transfer_functions(subject, condition, kc=kc)

    mag_d, phase_d = magnitude_and_phase(L_d)
    mag_t, phase_t = magnitude_and_phase(L_t)

    w_d_s, mag_d_s = smooth_curve_with_polynomial_regression(w, mag_d)
    _, phase_d_s = smooth_curve_with_polynomial_regression(w, phase_d)

    w_t_s, mag_t_s = smooth_curve_with_polynomial_regression(w, mag_t)
    _, phase_t_s = smooth_curve_with_polynomial_regression(w, phase_t)

    margins_d = compute_margins(w, L_d)
    margins_t = compute_margins(w, L_t)

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.2))
    fig.subplots_adjust(wspace=0.30, hspace=0.42)

    # a) Disturbance magnitude
    ax = axs[0, 0]
    setup_bode_axis(ax, r"$|H_{ol,d}|$", phase_plot=False)
    ax.plot(w_d_s, mag_d_s, "k-", linewidth=1.4, label="MLE")
    ax.plot(w, mag_d, linestyle="none", marker=r"$\ast$", markersize=9, color="k", label="FC")
    if np.isfinite(margins_d["gain_crossover_rad_s"]):
        ax.axvline(margins_d["gain_crossover_rad_s"], color="0.45", linewidth=0.8)
        ax.text(
            0.28,
            0.83,
            rf"$\omega_{{c,d}} = {margins_d['gain_crossover_rad_s']:.2f}\ \mathrm{{rad\ s^{{-1}}}}$",
            transform=ax.transAxes,
            fontsize=11,
        )
    ax.relim()
    ax.autoscale_view()
    ax.set_title("a) Disturbance magnitude", y=-0.33, fontweight="bold")

    # b) Target magnitude
    ax = axs[0, 1]
    setup_bode_axis(ax, r"$|H_{ol,t}|$", phase_plot=False)
    ax.plot(w_t_s, mag_t_s, "k-", linewidth=1.4)
    ax.plot(w, mag_t, linestyle="none", marker=r"$\ast$", markersize=9, color="k")
    if np.isfinite(margins_t["gain_crossover_rad_s"]):
        ax.axvline(margins_t["gain_crossover_rad_s"], color="0.45", linewidth=0.8)
        ax.text(
            0.28,
            0.83,
            rf"$\omega_{{c,t}} = {margins_t['gain_crossover_rad_s']:.2f}\ \mathrm{{rad\ s^{{-1}}}}$",
            transform=ax.transAxes,
            fontsize=11,
        )
    ax.relim()
    ax.autoscale_view()
    ax.set_title("b) Target magnitude", y=-0.33, fontweight="bold")

    # c) Disturbance phase
    ax = axs[1, 0]
    setup_bode_axis(ax, r"$\angle H_{ol,d},\ \mathrm{deg}$", phase_plot=True)
    ax.plot(w_d_s, phase_d_s, "k-", linewidth=1.4, label="MLE")
    ax.plot(w, phase_d, linestyle="none", marker=r"$\ast$", markersize=9, color="k", label="FC")
    draw_margin_annotation(ax, margins_d, mode="disturbance")
    ax.relim()
    ax.autoscale_view()
    ax.set_title("c) Disturbance phase", y=-0.33, fontweight="bold")

    # d) Target phase
    ax = axs[1, 1]
    setup_bode_axis(ax, r"$\angle H_{ol,t},\ \mathrm{deg}$", phase_plot=True)
    ax.plot(w_t_s, phase_t_s, "k-", linewidth=1.4, label="MLE")
    ax.plot(w, phase_t, linestyle="none", marker=r"$\ast$", markersize=9, color="k", label="FC")
    draw_margin_annotation(ax, margins_t, mode="target")
    ax.relim()
    ax.autoscale_view()
    ax.legend(
        loc="lower left",
        frameon=True,
        fancybox=False,
        edgecolor="k",
        borderpad=0.2,
        handlelength=1.0,
        handletextpad=0.35,
    )
    ax.set_title("d) Target phase", y=-0.33, fontweight="bold")

    fig.suptitle(f"Subject {subject} — {condition_name(condition)}", y=0.98, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if SAVE_FIGURES:
        outpath = outdir / f"subject_{subject:02d}_condition_{condition}_bode.png"
        fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    summary_row = {
        "subject": subject,
        "condition": condition,
        "condition_name": condition_name(condition),
        "disturbance_gain_crossover_rad_s": margins_d["gain_crossover_rad_s"],
        "disturbance_phase_margin_deg": margins_d["phase_margin_deg"],
        "disturbance_phase_crossover_rad_s": margins_d["phase_crossover_rad_s"],
        "disturbance_gain_margin": margins_d["gain_margin"],
        "target_gain_crossover_rad_s": margins_t["gain_crossover_rad_s"],
        "target_phase_margin_deg": margins_t["phase_margin_deg"],
        "target_phase_crossover_rad_s": margins_t["phase_crossover_rad_s"],
        "target_gain_margin": margins_t["gain_margin"],
    }

    return summary_row

# =============================================================================
# Run all subjects / conditions
# =============================================================================
def run_all(subjects=None, conditions=None, kc: float = 1.0):
    rows = []

    if subjects is None:
        subjects = sorted(dataset.keys())

    for subject in subjects:
        available_conditions = sorted(dataset[subject].keys())

        if conditions is None:
            condition_list = available_conditions
        else:
            condition_list = [c for c in conditions if c in available_conditions]

        for condition in condition_list:
            try:
                row = plot_subject_condition(subject, condition, kc=kc)
                rows.append(row)
                print(f"Done: subject={subject}, condition={condition}")
            except Exception as exc:
                print(f"Skipped subject={subject}, condition={condition}: {exc}")

    df = pd.DataFrame(rows).sort_values(["subject", "condition"]).reset_index(drop=True)

    csv_path = OUTPUT_DIR / "open_loop_margins.csv"
    xlsx_path = OUTPUT_DIR / "open_loop_margins.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    return df


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    """
    Before running:
    1. In Datasetcode.py, set:
           folder = r"path/to/your/mat/files"
    2. Install python-control:
           pip install control
    """

    summary = run_all(kc=Kc)

    print("\nSaved files:")
    print(OUTPUT_DIR / "open_loop_margins.csv")
    print(OUTPUT_DIR / "open_loop_margins.xlsx")
    print("\nFirst rows of summary:")
    print(summary.head())