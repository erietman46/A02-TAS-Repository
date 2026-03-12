import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
# from control.matlab import *
from Datasetcode import dataset

'''
TIME HISTORIES 
• the error signal                  e [deg]   8192x5 
• the control signal                u [deg]   8192x5  
• the controlled yaw angle          x [deg]   8192x5 
• the target signal                 ft [deg]   8192x1 
• the disturbance signal            fd [deg]  8192x1 
• the time vector                   t [s]   8192x1 
 
MEASURED PILOT FREQUENCY RESPONSES 
• the Hpe (visual) frequency response Hpe_FC [complex numbers] 20x1 
• the Hpxd (motion) frequency response  Hpxd_FC [complex numbers] 20x1 
• the frequency vector   w_FC [rad/s]   20x1    
'''

'''
• C1 = Gain (P), no motion 
• C2 = Single integrator (V), no motion 
• C3 = Double integrator (A), no motion 
• C4 = Gain (P), motion 
• C5 = Single integrator (V), motion 
• C6 = Double integrator (A), motion
'''

#__________________________________________________
## BODE PLOTS FOR PILOT RESPONSES
#__________________________________________________

# loop for pilot
for i in range(1,7):

# loop for experiments
    for j in range(1,7):
        
        #Absolute values of pilot responses
        H_pe_abs = abs(dataset[i][j]["Hpe_FC"])
        Hpxd_abs = abs(dataset[i][j]["Hpxd_FC"])

        #Phase angles of pilot responses
        H_pe_ang = np.angle(dataset[i][j]["Hpe_FC"], deg=True)
        Hpxd_ang = np.angle(dataset[i][j]["Hpxd_FC"], deg=True)

        #Unwrap data
        H_pe_ang = np.unwrap(H_pe_ang, period=360, axis=0)
        Hpxd_ang = np.unwrap(Hpxd_ang, period=360, axis=0)

        
        #Convert to decibels

        H_pe_db = 20 * np.log10(H_pe_abs)
        Hpxd_db = 20 * np.log10(Hpxd_abs)


        #Pilot response frequencies
        w_FC = dataset[i][j]["w_FC"]
    
        plt.figure(figsize=(10, 4))

        # Magnitude plot
        plt.subplot(1, 2, 1)
        plt.semilogx(w_FC, H_pe_db, 'o', label="Hpe")
        plt.semilogx(w_FC, Hpxd_db, 's', label="Hpxd")
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("Magnitude [dB]")
        plt.title(f"Subject {i}, Condition {j}")
        plt.grid(True, which="both")
        plt.legend()

        # Phase plot
        plt.subplot(1, 2, 2)
        plt.semilogx(w_FC, H_pe_ang, 'o', label="Hpe")
        plt.semilogx(w_FC, Hpxd_ang, 's', label="Hpxd")
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("Phase [deg]")
        plt.title(f"Subject {i}, Condition {j}")
        plt.grid(True, which="both")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"FIGURES/Subject {i}, Condition {j}")
        plt.close()
        # plt.show()


#Code to be able to call a single pilot and condition Bode plot
def plot_single_pilot_and_conditon(i,j):
        
        #Absolute values of pilot responses
        H_pe_abs = abs(dataset[i][j]["Hpe_FC"])
        Hpxd_abs = abs(dataset[i][j]["Hpxd_FC"])

        #Phase angles of pilot responses
        H_pe_ang = np.angle(dataset[i][j]["Hpe_FC"], deg=True)
        Hpxd_ang = np.angle(dataset[i][j]["Hpxd_FC"], deg=True)

        #Unwrap data
        H_pe_ang = np.unwrap(H_pe_ang, period=360, axis=0)
        Hpxd_ang = np.unwrap(Hpxd_ang, period=360, axis=0)

        
        #Convert to decibels

        H_pe_db = 20 * np.log10(H_pe_abs)
        Hpxd_db = 20 * np.log10(Hpxd_abs)


        #Pilot response frequencies
        w_FC = dataset[i][j]["w_FC"]
    
        plt.figure(figsize=(10, 4))

        # Magnitude plot
        plt.subplot(1, 2, 1)
        plt.semilogx(w_FC, H_pe_db, 'o', label="Hpe")
        plt.semilogx(w_FC, Hpxd_db, 's', label="Hpxd")
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("Magnitude [dB]")
        plt.title(f"Subject {i}, Condition {j}")
        plt.grid(True, which="both")
        plt.legend()

        # Phase plot
        plt.subplot(1, 2, 2)
        plt.semilogx(w_FC, H_pe_ang, 'o', label="Hpe")
        plt.semilogx(w_FC, Hpxd_ang, 's', label="Hpxd")
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("Phase [deg]")
        plt.title(f"Subject {i}, Condition {j}")
        plt.grid(True, which="both")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"FIGURES/Subject {i}, Condition {j}")
        plt.close()
        # plt.show()