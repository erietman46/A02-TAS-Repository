from preprocessing import pilots
import numpy as np

errors = np.zeros(30)

#Condition index: x
x = 0

#To calculate the mean square error for all conditions of all pilots and take the average of the their tests.
#P V A no motion then motion 1 --> 6
for i in range(len(pilots)):
    #looping pilots
    pilot = pilots[i]

    for j in range(pilot):
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

        MSE = sum/5
        errors[x] = MSE
        x += 1

print(errors)