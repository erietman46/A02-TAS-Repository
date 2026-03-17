import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

combinations = (("e", "u"), ("e", "u", "de"), ("e", "u", "du"), ("e", "u", "de", "du"))
