import numpy as np

def total_variation(input):
    total_variation = np.sum(np.abs(np.diff(input)))
    return total_variation