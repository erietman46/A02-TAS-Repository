from preprocessing import pilots
import stats
import numpy as np
import pandas as pd


def RMS_input():
    #Array to store RMS u
    u = np.zeros((6,6))


    #RMS will be calculated for all experiments and the average will be taken per condition per pilot.
    #P V A no motion then motion C1 --> C6
    for i in range(len(pilots)):
        #looping pilots
        pilot = pilots[i]

        for j in range(len(pilot)):
            #Condition C"k"
            #looping conditions
            k = j+1
            condition = f"C{k}"
            u_0 = pilot[condition]["u"]

            sum = 0

            for l in range(5):
                #looping columns
                column = u_0[:,l]
                mean_squared = np.mean(column**2)
                sum += np.sqrt(mean_squared)

            RMS = sum/5
            u[i,j] = RMS
        
    #An array which the rows are pilots and columns are conditions
    df = pd.DataFrame(
        u,
        index=[f"Pilot {i + 1}" for i in range(6)],
        columns=[f"C{i + 1}" for i in range(6)]
    )
    res = stats.statistics(np.array(u))
    results = pd.DataFrame(
        res,
        index=[f"C{i + 1}" for i in range(3)],
        columns=["p_val", "effect_size"]
    )

    return df, results, np.array(u)