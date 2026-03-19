# call all over functions, make everything look nice, ...
from zero_crossings import zero_crossings
from total_variation import total_variation
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

'''Zero-crossings of e'''
data, stat_res = zero_crossings()
print(f"\nMean Zero-crossings: \n{data}")
print(f"\nStatistical Results: \n{stat_res}")

'''Total Variation of u'''
data2, stat_res2 = total_variation()
print(f"\nMean Total Variation: \n{data2}")
print(f"\nStatistical Results: \n{stat_res2}")