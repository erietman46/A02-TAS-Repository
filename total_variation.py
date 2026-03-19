from preprocessing import pilots
import statistics
import numpy as np
import pandas as pd

def total_variation():
    variation = [[0] * 6 for _ in range(6)]
    for index, pilot in enumerate(pilots):
        for index2, condition in enumerate(pilot.values()):
            u = condition["u"]
            u = np.array(u)
            u_transposed = u.T
            total_variation = 0
            for i in u_transposed:
                total_variation += np.sum(np.abs(np.diff(i)))
            variation[index][index2] = total_variation / 5

    df = pd.DataFrame(
        variation,
        index=[f"Pilot {i + 1}" for i in range(6)],
        columns=[f"C{i + 1}" for i in range(6)]
    )
    res = statistics.statistics(np.array(variation))
    results = pd.DataFrame(
        res,
        index=[f"C{i + 1}" for i in range(3)],
        columns=["p_val", "effect_size"]
    )

    return df, results