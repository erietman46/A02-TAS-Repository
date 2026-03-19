from preprocessing import pilots
import statistics
import numpy as np
import pandas as pd

def zero_crossings():
    crossings = [[0] * 6 for _ in range(6)]
    for index, pilot in enumerate(pilots):
        for index2, condition in enumerate(pilot.values()):
            e = condition["e"]
            e = np.array(e)
            e_transposed = e.T
            cross = 0
            for i in e_transposed:
                for j in range(len(i) - 1):
                    current = i[j]
                    next = i[j + 1]
                    if current * next < 0:
                        cross += 1
            crossings[index][index2] = cross / 5

    df = pd.DataFrame(
        crossings,
        index=[f"Pilot {i + 1}" for i in range(6)],
        columns=[f"C{i + 1}" for i in range(6)]
    )
    res = statistics.statistics(np.array(crossings))
    results = pd.DataFrame(
        res,
        index=[f"C{i + 1}" for i in range(3)],
        columns=["p_val", "effect_size"]
    )

    return df, results