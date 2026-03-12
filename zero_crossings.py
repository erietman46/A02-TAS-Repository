from preprocessing import pilots
import numpy as np

pilot1 = pilots[0]
C1 = pilot1["C1"]
e = C1["e"]
e = np.array(e)
e_transposed = e.T

crossings = 0
for index, i in enumerate(e_transposed):
    for j in range(len(i) - 1):
        current = i[j]
        next = i[j + 1]
        if current * next < 0:
            crossings += 1
crossings = crossings/5

print(crossings)