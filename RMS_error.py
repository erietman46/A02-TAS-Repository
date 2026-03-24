from preprocessing import pilots
import statistics
import numpy as np
import pandas as pd

#Array to store RMS errors
errors = np.zeros((6,6))


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
        error = pilot[condition]["e"]

        sum = 0

        for l in range(5):
            #looping columns
            column = error[:,l]
            mean_squared = np.mean(column**2)
            sum += np.sqrt(mean_squared)

        RMS = sum/5
        errors[i,j] = RMS
        
#An array which the rows are pilots and columns are conditions
df = pd.DataFrame(
    errors,
    index=[f"Pilot {i+1}" for i in range(6)],
    columns=[f"C{i+1}" for i in range(6)]
)
print(f"\nMean RMS: \n{df}")
res = statistics.statistics(np.array(errors))
results = pd.DataFrame(
    res,
    index=[f"C{i+1}" for i in range(3)],
    columns=["p_val", "effect_size"]
)
print(f"\nStatistical Results: \n{results}")