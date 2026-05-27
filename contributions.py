import numpy as np
import matplotlib.pyplot as plt
from preprocessing import pilots
import pandas as pd
import stats

def helper(u):

    # ==============================
    # INPUT: your data
    # ==============================
    # u should be either:
    # shape (8192, 5) OR list of 5 arrays of length 8192

    u = np.array(u)

    # ensure shape = (n_points, n_runs)
    if u.shape[0] == 5:
        u = u.T

    n_points, n_runs = u.shape

    # ==============================
    # CONSTANTS
    # ==============================
    dt = 1 / 100  # sampling time (100 Hz)
    dw = 2 * np.pi / (n_points * dt)

    # MATLAB indices → Python indices
    nd = np.array([5, 11, 23, 37, 51, 71, 101, 137, 171, 226]) - 1
    nt = np.array([6, 13, 27, 41, 53, 73, 103, 139, 194, 229]) - 1

    # ==============================
    # STORAGE
    # ==============================
    var_fd = np.zeros(n_runs)
    var_ft = np.zeros(n_runs)
    var_noise = np.zeros(n_runs)
    var_total = np.zeros(n_runs)
    var_time = np.zeros(n_runs)

    # ==============================
    # MAIN LOOP
    # ==============================
    for rn in range(n_runs):
        v = u[:, rn]

        # FFT
        V = np.fft.fft(v)
        V = V[1:n_points // 2 + 1]  # positive frequencies only

        # disturbance contribution
        var_fd[rn] = np.sum((np.abs(V[nd]) / (n_points / 2)) ** 2) / 2

        # target contribution
        var_ft[rn] = np.sum((np.abs(V[nt]) / (n_points / 2)) ** 2) / 2

        # total variance (Parseval)
        total = np.sum(dw * (V * np.conj(V)) / n_points * dt) / np.pi
        var_total[rn] = np.real(total)

        # noise = remainder
        var_noise[rn] = var_total[rn] - var_fd[rn] - var_ft[rn]

        # time-domain check
        var_time[rn] = np.var(v)

    # # ==============================
    # # PRINT RESULTS
    # # ==============================
    # print("\nPer run:")
    # for i in range(n_runs):
    #     print(
    #         f"Run {i + 1}: fd={var_fd[i]:.4f}, ft={var_ft[i]:.4f}, noise={var_noise[i]:.4f}, total={var_total[i]:.4f}")

    # ==============================
    # AVERAGES
    # ==============================
    mean_fd = np.mean(var_fd)
    mean_ft = np.mean(var_ft)
    mean_noise = np.mean(var_noise)

    total_mean = mean_fd + mean_ft + mean_noise

    mean_per_fd = 100 * mean_fd / total_mean
    mean_per_ft = 100 * mean_ft / total_mean
    mean_per_noise = 100 * mean_noise / total_mean

    # print("\nAverages:")
    # print(f"Disturbance: {mean_fd:.4f} ({100 * mean_fd / total_mean:.1f}%)")
    # print(f"Target:      {mean_ft:.4f} ({100 * mean_ft / total_mean:.1f}%)")
    # print(f"Noise:       {mean_noise:.4f} ({100 * mean_noise / total_mean:.1f}%)")
    #
    # # ==============================
    # # STACKED BAR PLOT
    # # ==============================
    # varS = np.vstack((var_fd, var_ft, var_noise)).T
    #
    # plt.figure()
    # plt.bar(range(1, n_runs + 1), varS[:, 0], label="f_d")
    # plt.bar(range(1, n_runs + 1), varS[:, 1], bottom=varS[:, 0], label="f_t")
    # plt.bar(range(1, n_runs + 1), varS[:, 2], bottom=varS[:, 0] + varS[:, 1], label="noise")
    #
    # plt.xlabel("Run #")
    # plt.ylabel("Variance of u")
    # plt.title("Variance Decomposition of Pilot Input")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # ==============================
    # OPTIONAL: sanity check
    # ==============================
    print("\nTime vs frequency domain variance check:")
    for i in range(n_runs):
        print(f"Run {i + 1}: time={var_time[i]:.4f}, freq={var_total[i]:.4f}")

    return mean_per_fd, mean_per_ft, mean_per_noise

def contributions():
    metric_disturbance = [[0] * 6 for _ in range(6)]
    metric_target = [[0] * 6 for _ in range(6)]
    metric_noise = [[0] * 6 for _ in range(6)]
    for index, pilot in enumerate(pilots):
        for index2, condition in enumerate(pilot.values()):
            u = condition["u"]
            u = np.array(u)
            mean_fd, mean_ft, mean_noise = helper(u)
            metric_disturbance[index][index2] = float(mean_fd)
            metric_target[index][index2] = float(mean_ft)
            metric_noise[index][index2] = float(mean_noise)
    df1 = pd.DataFrame(
        metric_disturbance,
        index=[f"Pilot {i + 1}" for i in range(6)],
        columns=[f"C{i + 1}" for i in range(6)]
    )
    res1 = stats.statistics(np.array(metric_disturbance))
    results1 = pd.DataFrame(
        res1,
        index=[f"C{i + 1}" for i in range(3)],
        columns=["p_val", "effect_size", "effect_type"]
    )
    df2 = pd.DataFrame(
        metric_target,
        index=[f"Pilot {i + 1}" for i in range(6)],
        columns=[f"C{i + 1}" for i in range(6)]
    )
    res2 = stats.statistics(np.array(metric_target))
    results2 = pd.DataFrame(
        res2,
        index=[f"C{i + 1}" for i in range(3)],
        columns=["p_val", "effect_size", "effect_type"]
    )
    df3 = pd.DataFrame(
        metric_noise,
        index=[f"Pilot {i + 1}" for i in range(6)],
        columns=[f"C{i + 1}" for i in range(6)]
    )
    res3 = stats.statistics(np.array(metric_noise))
    results3 = pd.DataFrame(
        res3,
        index=[f"C{i + 1}" for i in range(3)],
        columns=["p_val", "effect_size", "effect_type"]
    )

    return df1, results1, df2, results2, df3, results3, np.array(metric_disturbance), np.array(metric_target), np.array(metric_noise)

helper(np.array(pilots[2]["C6"]["u"]))