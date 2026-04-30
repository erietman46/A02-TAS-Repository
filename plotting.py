# =============================================================================
# PLOTTING STYLE GUIDE
# How to convert a basic matplotlib Bode plot to the project style
# =============================================================================
#
# This file walks through each adjustment needed, step by step.
# Each section shows the OLD approach in a comment, followed by the
# NEW code that replaces it.
# =============================================================================


import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# STEP 1 — SET GLOBAL RCPARAMS
# =============================================================================
#
# A plain script uses matplotlib's default fonts and sizes.
# Replace the defaults by calling plt.rcParams.update() once, near the top
# of your file, before any figures are created.
#
# OLD (implicit defaults):
#   import matplotlib.pyplot as plt   # nothing else needed — but output looks generic
#
# NEW:

plt.rcParams.update({
    "font.family":        "serif",
    "mathtext.fontset":   "stix",       # LaTeX-style math glyphs
    "font.size":          11,
    "axes.labelsize":     14,
    "axes.titlesize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    10,
    "axes.linewidth":     0.8,
    "grid.linewidth":     0.7,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

# This affects every figure created after this point in the script.


# =============================================================================
# STEP 2 — SET SAVE DPI
# =============================================================================
#
# OLD:
#   fig.savefig("output.png", dpi=200)
#
# NEW — define a constant and use it in every savefig call:

FIG_DPI = 220


# =============================================================================
# STEP 3 — CHANGE bode_mag_phase TO RETURN ABSOLUTE MAGNITUDE
# =============================================================================
#
# OLD — magnitude in decibels, phase via np.angle(..., deg=True):
#
#   def bode_mag_phase(H):
#       H_db  = 20 * np.log10(np.abs(H))
#       H_ang = np.angle(H, deg=True)
#       H_ang = np.unwrap(H_ang, period=360, axis=0)
#       return H_db, H_ang
#
# NEW — absolute magnitude, phase via np.unwrap then convert to degrees.
# Absolute magnitude is needed so the y-axis can be set to log scale (Step 5).

def bode_mag_phase(H):
    mag       = np.abs(H)
    phase_deg = np.unwrap(np.angle(H)) * 180.0 / np.pi
    return mag, phase_deg


# =============================================================================
# STEP 4 — ADD A setup_bode_axis HELPER
# =============================================================================
#
# OLD — axis formatting was repeated inline in every plot function:
#
#   axes[0].grid(True, which="both")
#   axes[0].set_xlabel("Frequency [rad/s]")
#   axes[0].set_ylabel("Magnitude [dB]")
#
# NEW — one reusable helper that applies the full style to any axis.
# Pass phase_plot=True for the phase subplot, False for the magnitude subplot.

def setup_bode_axis(ax, ylabel: str, phase_plot: bool = False):
    # Log scale on x-axis (frequency axis)
    ax.set_xscale("log")

    # Enable minor tick marks so the minor grid has something to follow
    ax.minorticks_on()

    # Two-tier grid: solid major lines, dotted minor lines, both in grey
    ax.grid(True, which="major", color="0.70")
    ax.grid(True, which="minor", color="0.70", linestyle=(0, (1.2, 4.5)))

    # LaTeX frequency label with correct units
    ax.set_xlabel(r"$\omega,\ \mathrm{rad\ s^{-1}}$")
    ax.set_ylabel(ylabel)

    # Square aspect ratio for each subplot
    ax.set_box_aspect(1)

    if phase_plot:
        # Horizontal reference line at -180 deg (stability boundary)
        ax.axhline(-180, color="0.4", linewidth=0.8)
    else:
        # Log y-scale for magnitude; reference line at unity gain
        ax.set_yscale("log")
        ax.axhline(1.0, color="0.4", linewidth=0.8)


# =============================================================================
# STEP 5 — CHANGE MARKER AND LINE STYLE
# =============================================================================
#
# OLD — circle/square markers connected to the fit line:
#
#   axes[0].semilogx(w, data_db, 'o', label="Measured")
#   axes[0].semilogx(w, fit_db,  '-', label="Fitted")
#
# NEW — star markers for measured data (no connecting line), solid black line
# for the fit. Use ax.plot() instead of ax.semilogx() because the log x-scale
# is now set by setup_bode_axis.

# Measured data points:
#   ax.plot(w, mag, linestyle="none", marker=r"$\ast$", markersize=9, color="k", label="...")

# Fitted curve:
#   ax.plot(w, fit_mag, "k-", linewidth=1.4, label="...")


# =============================================================================
# STEP 6 — RESTRUCTURE FIGURE LAYOUT (1×2 → 2×1)
# =============================================================================
#
# OLD — magnitude and phase side by side:
#
#   fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#
# NEW — magnitude on top, phase below. This is the standard Bode layout and
# matches how setup_bode_axis labels the y-axes.

fig, axes = plt.subplots(2, 1, figsize=(5.4, 7.2))
fig.subplots_adjust(hspace=0.42)
plt.close(fig)  # close the demo figure; real usage continues below


# =============================================================================
# STEP 7 — MOVE SUBPLOT TITLES BELOW THE AXES
# =============================================================================
#
# OLD — title above the subplot (matplotlib default, y=1.0):
#
#   ax.set_title(f"Visual Bode Plot - Subject {i}, Condition {j}")
#
# NEW — title below the subplot, bold:

#   ax.set_title("a) Visual magnitude", y=-0.33, fontweight="bold")
#   ax.set_title("b) Visual phase",     y=-0.33, fontweight="bold")

# Use letter prefixes (a, b, ...) to label each panel.


# =============================================================================
# STEP 8 — ADD fig.suptitle AND UPDATE tight_layout
# =============================================================================
#
# OLD:
#   fig.tight_layout()
#
# NEW — add a centred figure title and reserve space for it:

#   fig.suptitle(f"Subject {i} — Condition {j} — Visual ($H_{{pe}}$)", y=0.98, fontsize=13)
#   fig.tight_layout(rect=[0, 0, 1, 0.97])


# =============================================================================
# STEP 9 — APPLY LEGEND STYLE
# =============================================================================
#
# OLD:
#   ax.legend()
#
# NEW — place the legend only on the phase subplot (bottom panel), styled to
# match the rest of the figure:

#   ax.legend(
#       loc="lower left",
#       frameon=True,
#       fancybox=False,
#       edgecolor="k",
#       borderpad=0.2,
#       handlelength=1.0,
#       handletextpad=0.35,
#   )


# =============================================================================
# COMPLETE EXAMPLE — putting it all together
# =============================================================================
#
# Below is what a full save_visual_bode function looks like after all changes.

def save_visual_bode(i, j, w_FC, vis_data, visual_fit):
    vis_mag,  vis_ang  = bode_mag_phase(vis_data)
    fit_mag,  fit_ang  = bode_mag_phase(visual_fit)

    fig, axes = plt.subplots(2, 1, figsize=(5.4, 7.2))
    fig.subplots_adjust(hspace=0.42)

    # — Magnitude subplot —
    ax = axes[0]
    setup_bode_axis(ax, r"$|H_{pe}|$", phase_plot=False)
    ax.plot(w_FC, fit_mag, "k-", linewidth=1.4,  label="Fitted $H_{pe}$")
    ax.plot(w_FC, vis_mag, linestyle="none", marker=r"$\ast$",
            markersize=9, color="k",             label="Measured $H_{pe}$")
    ax.relim()
    ax.autoscale_view()
    ax.set_title("a) Visual magnitude", y=-0.33, fontweight="bold")

    # — Phase subplot —
    ax = axes[1]
    setup_bode_axis(ax, r"$\angle H_{pe},\ \mathrm{deg}$", phase_plot=True)
    ax.plot(w_FC, fit_ang, "k-", linewidth=1.4,  label="Fitted $H_{pe}$")
    ax.plot(w_FC, vis_ang, linestyle="none", marker=r"$\ast$",
            markersize=9, color="k",             label="Measured $H_{pe}$")
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
    ax.set_title("b) Visual phase", y=-0.33, fontweight="bold")

    fig.suptitle(
        f"Subject {i} \u2014 Condition {j} \u2014 Visual ($H_{{pe}}$)",
        y=0.98, fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        f"FIGURES/subject_{i}_condition_{j}_visual.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
