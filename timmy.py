#fuck you yoshua
#no :(
#import scipy.io as sc
#import matplotlib.pyplot as plt 


data = sc.loadmat("ae2224I_measurement_data_subj1_C1.mat")
t    = data["t"][0]
e    = data["e"]
e1   = e[:,0]


plt.plot(t,e1)
plt.show()