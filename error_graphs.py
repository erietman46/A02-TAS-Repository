from preprocessing import pilots
import matplotlib.pyplot as plt
import numpy as np


fig, axs = plt.subplots(6,6)


for i in range(len(pilots)):
    #looping pilots
    pilot = pilots[i]

    for j in range(len(pilot)):
        #Condition C"k"
        #looping conditions
        k = j+1
        condition = f"C{k}"
        error = pilot[condition]["e"][:,2]
        time = pilot[condition]["t"].T
        axs[i,j].plot(time,error)


plt.subplots_adjust(wspace=0.275, hspace=0.275, left= 0.05, bottom= 0.05, right= 0.95, top= 0.95)
plt.show()