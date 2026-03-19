import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
# from control.matlab import *
from Datasetcode import dataset
import Optimizedpilotfitting_official as opf


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


def bode_mag_phase(H):
    """Return magnitude in dB and unwrapped phase in degrees."""
    H_abs = np.abs(H)
    H_db = 20 * np.log10(H_abs)
    H_ang = np.angle(H, deg=True)
    H_ang = np.unwrap(H_ang, period=360, axis=0)
    return H_db, H_ang


costs = []
parameters_C1 = np.zeros((6,13))
parameters_C2 = np.zeros((6,13))
parameters_C3 = np.zeros((6,13))
parameters_C4 = np.zeros((6,13))
parameters_C5 = np.zeros((6,13))
parameters_C6 = np.zeros((6,13))
global_parameters = [parameters_C1, parameters_C2, parameters_C3, parameters_C4, parameters_C5, parameters_C6]

# loop for pilot
for i in range(1,7):

# loop for experiments
    for j in range(1,7):
        

        motion = j in [4,5,6]



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

        #Try fitted models
        #visual_fit, vestib_fit, result, cost  = opf.fit_subject_condition(i, j)
        visual_fit, vestib_fit, best_result, best_cost = opf.fit_subject_condition(i, j)
        cost = best_cost
        visual_fit_db, visual_fit_ang = bode_mag_phase(visual_fit)

        #Append costs to list
        costs.append(cost)

        #Add parameters to list
        params = best_result.x
        while len(params) < 13:
             params = np.append(params, 0)
        
        global_parameters[j-1][i-1] = params
             

        #Visual parameters: Kp, TL, TI, tau, omega_nm, zeta_nm
        # Vestibular parameters: Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm, zeta_nm


        # ==========================================================
        # VISUAL BODE PLOT
        # ==========================================================

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.semilogx(w_FC, H_pe_db, 'o', label="Measured Hpe")
        plt.semilogx(w_FC, visual_fit_db, '-', label="Fitted Hpe")
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("Magnitude [dB]")
        plt.title(f"Visual Bode Plot - Subject {i}, Condition {j}")
        plt.grid(True, which="both")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.semilogx(w_FC, H_pe_ang, 'o', label="Measured Hpe")
        plt.semilogx(w_FC, visual_fit_ang, '-', label="Fitted Hpe")
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("Phase [deg]")
        plt.title(f"Visual Bode Plot - Subject {i}, Condition {j}")
        plt.grid(True, which="both")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"FIGURES/subject_{i}_condition_{j}_visual.png", dpi=200)
        plt.close()

        # ==========================================================
        # VESTIBULAR BODE PLOT (MOTION CONDITIONS ONLY)
        # ==========================================================
        if motion:
            Hpxd = dataset[i][j]["Hpxd_FC"]
            Hpxd_db, Hpxd_ang = bode_mag_phase(Hpxd)

            vestib_fit_db, vest_fit_ang = bode_mag_phase(vestib_fit)

            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.semilogx(w_FC, Hpxd_db, 's', label="Measured Hpxd")
            plt.semilogx(w_FC, vestib_fit_db, '-', label="Fitted Hpxd")
            plt.xlabel("Frequency [rad/s]")
            plt.ylabel("Magnitude [dB]")
            plt.title(f"Vestibular Bode Plot - Subject {i}, Condition {j}")
            plt.grid(True, which="both")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.semilogx(w_FC, Hpxd_ang, 's', label="Measured Hpxd")
            plt.semilogx(w_FC, vest_fit_ang, '-', label="Fitted Hpxd")
            plt.xlabel("Frequency [rad/s]")
            plt.ylabel("Phase [deg]")
            plt.title(f"Vestibular Bode Plot - Subject {i}, Condition {j}")
            plt.grid(True, which="both")
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"FIGURES/subject_{i}_condition_{j}_vestibular.png", dpi=200)
            plt.close()

        print(f"Finished Subject {i}, Condition {j}, Cost = {cost:.4f}")

        """""
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

        """




#Print costs summary
npcosts = np.array(costs)
np.save("global_parameters.npy", np.array(global_parameters, dtype=object))
print('mean:', np.mean(npcosts), '\n max:', np.max(npcosts), '\n min', np.min(npcosts))


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

