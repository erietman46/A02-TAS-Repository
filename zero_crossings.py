from preprocessing import pilots
import numpy as np

crossings = [[0]*6 for _ in range(6)]
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
        crossings[index][index2] = cross/5

print(crossings)