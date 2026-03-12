from preprocessing import pilots
import numpy as np

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
        error = pilot[condition]["u"]

        sum = 0

        for l in range(5):
            #looping columns
            column = error[:,l]
            mean_squared = np.mean(column**2)
            sum += np.sqrt(mean_squared)

        MSE = sum/5
        u[i,j] = MSE
        
#An array which the rows are pilots and columns are conditions
print(u)