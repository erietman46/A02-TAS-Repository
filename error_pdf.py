from preprocessing import pilots
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import stats

def interval(errors):
    #determines the mean and the 1 sigma interval width
    mu = np.mean(errors)
    sigma = np.std(errors)
    return mu, 2*sigma

def visualization(errors):
    #define normal pdf to overlay and compare
    x = np.linspace(min(errors), max(errors), 200)
    mu = np.mean(errors)
    sigma = np.std(errors)
    pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    plt.hist(errors, bins=80, density=True)
    plt.xlabel("Tracking error")
    plt.ylabel("Probability density")
    plt.plot(x, pdf) #plot normal pdf
    plt.show()

def error_pdf():
    mus = [[0] * 6 for _ in range(6)]
    sigmas = [[0] * 6 for _ in range(6)]
    for index, pilot in enumerate(pilots):
        for index2, condition in enumerate(pilot.values()):
            e = condition["e"]
            e = np.array(e)
            e_transposed = e.T
            inter_mus = [0] * 5
            inter_sigmas = [0] * 5
            for index3, i in enumerate(e_transposed):
                mu, sigma = interval(i)
                inter_mus[index3] = mu
                inter_sigmas[index3] = sigma
            mus[index][index2] = np.mean(inter_mus)
            sigmas[index][index2] = np.mean(inter_sigmas)

    clean_mus = np.array(mus, dtype=float)
    clean_sigmas = np.array(sigmas, dtype=float)
    df = pd.DataFrame(
        clean_mus,
        index=[f"Pilot {i + 1}" for i in range(6)],
        columns=[f"C{i + 1}" for i in range(6)]
    )
    res = stats.statistics(np.array(clean_mus))
    results = pd.DataFrame(
        res,
        index=[f"C{i + 1}" for i in range(3)],
        columns=["p_val", "effect_size", "effect_type"]
    )
    df2 = pd.DataFrame(
        clean_sigmas,
        index=[f"Pilot {i + 1}" for i in range(6)],
        columns=[f"C{i + 1}" for i in range(6)]
    )
    res2 = stats.statistics(np.array(clean_sigmas))
    results2 = pd.DataFrame(
        res2,
        index=[f"C{i + 1}" for i in range(3)],
        columns=["p_val", "effect_size", "effect_type"]
    )

    return df, results, df2, results2, np.array(clean_mus), np.array(clean_sigmas)