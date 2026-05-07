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

OUTPUT_DIR = Path("open_loop_results")
OUTPUT_DIR.mkdir(exist_ok=True)

SAVE_FIGURES = True
SHOW_FIGURES = False
FIG_DPI = 220

POLY_DEGREE = 4
RIDGE_ALPHA = 1e-3
SMOOTH_POINTS = 500

USE_MEASURED_FC_DATA = True 

#plot style
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

def short_condition_label(condition: int) -> str:
    return f"C{condition}"

def is_motion_condition(condition: int) -> bool:
    return condition in [4, 5, 6]

def is_valid_condition(condition: int) -> bool:
    return condition in [1, 2, 3, 4, 5, 6]

def controlled_element_frf(w: np.ndarray, condition: int, kc: float = 1.0) -> np.ndarray:
    s = 1j * w

    if condition in [1, 4]:
        return kc * 1 / (0.1 * s + 1)**3

    if condition in [2, 5]:
        return kc * 4 / (s * (0.03333333 * s + 1)**2)

    if condition in [3, 6]:
        return kc * 15 / (s**2)

    raise ValueError("Condition must be one of [1, 2, 3, 4, 5, 6].")

def get_measured_frequency_data(subject: int, condition: int):
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

def compute_open_loop_transfer_functions(subject: int, condition: int, kc: float = 1.0):
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

def magnitude_and_phase(H: np.ndarray):
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

def find_unity_gain_crossings(w: np.ndarray, mag: np.ndarray):
    w = np.asarray(w).ravel().astype(float)
    mag = np.asarray(mag).ravel().astype(float)

    valid = np.isfinite(w) & np.isfinite(mag) & (w > 0) & (mag > 0)
    w = w[valid]
    mag = mag[valid]

    if len(w) < 2:
        return []

    x = np.log10(w)
    y = np.log10(mag)   # unity gain => y = 0

    crossings = []

    for i in range(len(y) - 1):
        y1, y2 = y[i], y[i + 1]

        if y1 == 0:
            crossings.append(w[i])
        elif y1 * y2 < 0:
            t = -y1 / (y2 - y1)
            x_cross = x[i] + t * (x[i + 1] - x[i])
            crossings.append(10 ** x_cross)

    if y[-1] == 0:
        crossings.append(w[-1])

    cleaned = []
    for wc in crossings:
        if not cleaned or abs(np.log10(wc) - np.log10(cleaned[-1])) > 1e-6:
            cleaned.append(wc)

    return cleaned

def interpolate_phase_at_frequency(w: np.ndarray, phase_deg: np.ndarray, wc: float):
    w = np.asarray(w).ravel().astype(float)
    phase_deg = np.asarray(phase_deg).ravel().astype(float)

    valid = np.isfinite(w) & np.isfinite(phase_deg) & (w > 0)
    w = w[valid]
    phase_deg = phase_deg[valid]

    if len(w) < 2 or not np.isfinite(wc) or wc <= 0:
        return np.nan

    x = np.log10(w)
    xc = np.log10(wc)

    return np.interp(xc, x, phase_deg)

def compute_margins(w: np.ndarray, L: np.ndarray, crossover="first"):

    mag, phase_deg = magnitude_and_phase(L)

    gain_crossings = find_unity_gain_crossings(w, mag)

    if len(gain_crossings) == 0:
        return {
            "gain_margin": np.nan,
            "phase_margin_deg": np.nan,
            "phase_crossover_rad_s": np.nan,
            "gain_crossover_rad_s": np.nan,
            "all_gain_crossovers_rad_s": [],
        }

    wc = gain_crossings[-1] if crossover == "last" else gain_crossings[0]
    phase_at_wc = interpolate_phase_at_frequency(w, phase_deg, wc)
    pm = 180.0 + phase_at_wc

    return {
        "gain_margin": np.nan,
        "phase_margin_deg": float(pm) if np.isfinite(pm) else np.nan,
        "phase_crossover_rad_s": np.nan,
        "gain_crossover_rad_s": float(wc),
        "all_gain_crossovers_rad_s": gain_crossings,
    }

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

        text = rf"$\varphi_{{m,d}} = {pm:.1f}^\circ$" if mode == "disturbance" else rf"$\varphi_{{m,t}} = {pm:.1f}^\circ$"
        ax.text(0.33, 0.27, text, transform=ax.transAxes, fontsize=12)

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

    if margins_d["phase_margin_deg"] > 360:
        margins_d["phase_margin_deg"] = margins_d["phase_margin_deg"] - 360
        phase_d = phase_d - 360
        _, phase_d_s = smooth_curve_with_polynomial_regression(w, phase_d)
    
    if margins_t["phase_margin_deg"] > 360:
        margins_t["phase_margin_deg"] = margins_t["phase_margin_deg"] - 360
        phase_t = phase_t - 360
        _, phase_t_s = smooth_curve_with_polynomial_regression(w, phase_t)

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.2))
    fig.subplots_adjust(wspace=0.30, hspace=0.42)

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

    ax = axs[1, 0]
    setup_bode_axis(ax, r"$\angle H_{ol,d},\ \mathrm{deg}$", phase_plot=True)
    ax.plot(w_d_s, phase_d_s, "k-", linewidth=1.4, label="MLE")
    ax.plot(w, phase_d, linestyle="none", marker=r"$\ast$", markersize=9, color="k", label="FC")
    draw_margin_annotation(ax, margins_d, mode="disturbance")
    ax.relim()
    ax.autoscale_view()
    ax.set_title("c) Disturbance phase", y=-0.33, fontweight="bold")

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

def _group_values_by_condition(df: pd.DataFrame, value_col: str, conditions=(1, 2, 3, 4, 5, 6)):
    grouped = []
    for c in conditions:
        vals = df.loc[df["condition"] == c, value_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        grouped.append(vals)
    return grouped

def _style_boxplot(bp):
    for box in bp["boxes"]:
        box.set(color="k", linewidth=1.0)
    for whisker in bp["whiskers"]:
        whisker.set(color="k", linewidth=0.9)
    for cap in bp["caps"]:
        cap.set(color="k", linewidth=0.9)
    for median in bp["medians"]:
        median.set(color="k", linewidth=1.2)
    for flier in bp["fliers"]:
        flier.set(marker="o", markerfacecolor="white", markeredgecolor="k", markersize=4, linestyle="none")

def setup_boxplot_axis(ax, ylabel: str):
    ax.grid(True, axis="y", which="major", color="0.75")
    ax.grid(True, axis="y", which="minor", color="0.82", linestyle=(0, (1.2, 4.5)))
    ax.minorticks_on()
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Condition")
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([short_condition_label(c) for c in range(1, 7)])
    ax.set_box_aspect(1)

def plot_boxplots_all_conditions(df: pd.DataFrame, outdir: Path = OUTPUT_DIR):
    conditions_all = [1, 4, 2, 5, 3, 6]
    labels_all = [short_condition_label(c) for c in conditions_all]

    boxplot_specs = [
        ("target_gain_crossover_rad_s", r"Crossover frequency, $\omega_{c,t}$ [rad s$^{-1}$]", conditions_all, labels_all, "target_crossover_frequency"),
        ("disturbance_gain_crossover_rad_s", r"Crossover frequency, $\omega_{c,d}$ [rad s$^{-1}$]", conditions_all, labels_all, "disturbance_crossover_frequency"),
        ("target_phase_margin_deg", r"Phase margin, $\varphi_{m,t}$ [deg]", conditions_all, labels_all, "target_phase_margin"),
        ("disturbance_phase_margin_deg", r"Phase margin, $\varphi_{m,d}$ [deg]", conditions_all, labels_all, "disturbance_phase_margin"),
    ]

    for value_col, ylabel, conditions, labels, filename_stub in boxplot_specs:
        grouped_values = _group_values_by_condition(df, value_col, conditions)

        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        bp = ax.boxplot(grouped_values, widths=0.55, patch_artist=False, showfliers=True)
        _style_boxplot(bp)
        setup_boxplot_axis(ax, ylabel)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)

        fig.tight_layout()

        if SAVE_FIGURES:
            fig.savefig(outdir / f"boxplot_{filename_stub}_all_conditions.png", dpi=FIG_DPI, bbox_inches="tight")

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)

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

    plot_boxplots_all_conditions(df, outdir=OUTPUT_DIR)

    return df



if __name__ == "__main__":

    summary = run_all(kc=Kc)

    print("\nSaved files:")
    print(OUTPUT_DIR / "open_loop_margins.csv")
    print(OUTPUT_DIR / "open_loop_margins.xlsx")
    print(OUTPUT_DIR / "boxplot_crossover_frequency_all_conditions.png")
    print(OUTPUT_DIR / "boxplot_phase_margin_all_conditions.png")
    print("\nFirst rows of summary:")
    print(summary.head())