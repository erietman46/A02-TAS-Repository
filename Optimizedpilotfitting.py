import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from Datasetcode import dataset


# Definition of parameters and initialisation

w = 1           #
omega_nm = 1    #
zeta_Nm = 1     #

Kp = 1          #
TL = 1          #
TI = 1          #
tau = 1         #

Tsc1 = 1        #
Tsc2 = 1        #
Tsc3 = 1        #

Km = 1          #
tau_m = 1       #







#Neuromuscular model
def Hnm_model(w, omega_nm, zeta_nm):
    s = 1j * w
    return omega_nm**2 / (s**2 + 2*zeta_nm*omega_nm*s + omega_nm**2)

#MAIN MODEL: Visual pilot model
def Hpe_model(w, Kp, TL, TI, tau, omega_nm, zeta_nm):
    s = 1j * w

    equalization = (TL * s + 1) / (TI * s + 1)
    nm = Hnm_model(w, omega_nm, zeta_nm)
    delay = np.exp(-s * tau)

    return Kp * equalization * nm * delay


#Semicircular canal model
def Hsc_model(w, Tsc1, Tsc2, Tsc3):
    s = 1j * w
    return (1 + Tsc1 * s) / ((1 + Tsc2 * s) * (1 + Tsc3 * s))


# MAIN MODEL: Vestibular model
def Hpxd_model(w, Km, Tsc1, Tsc2, Tsc3, tau_m, omega_nm, zeta_nm):
    s = 1j * w

    sc = Hsc_model(w, Tsc1, Tsc2, Tsc3)
    nm = Hnm_model(w, omega_nm, zeta_nm)
    delay = np.exp(-s * tau_m)

    return Km * s * sc * delay * nm


# cost functions
def cost_function(cf_vis_data, cf_vis_model, cf_vest_data, cf_vest_model, weight_vis=1, weight_vest=1):
    cost = 0
    #Make sure that, when the models are called, the frequency matches the data
    for i in range(len(cf_vis_data)):
        cost += weight_vis*((cf_vis_data[i]-cf_vis_model)**2)/(cf_vis_data[i]**2) + weight_vest*((cf_vest_data[i]-cf_vest_model)**2)/(cf_vest_data[i]**2)
    return cost

def cost_function(w, cf_vis_data, cf_vs_data, vis_params, vest_params, weight_vis=1, weight_vest=1):
    """
    w: frequency array: Already given

    cf_vis_data: visual data array

    cf_vs_model: function that takes w and parameters and returns the model output

    vis_params: parameters for the visual model

    vest_params: parameters for the vestibular model
    """

    cost = 0

    cf_vis_model = Hpe_model(w, *vis_params)
    cf_vest_model = Hpxd_model(w, *vest_params)

    err_vis = np.abs(cf_vis_data - cf_vis_model)**2 / np.abs(cf_vis_data)**2
    err_vest = np.abs(cf_vs_data - cf_vest_model)**2 / np.abs(cf_vs_data)**2

    cost = weight_vis * np.sum(err_vis) + weight_vest * np.sum(err_vest)

    return float(np.real(cost))





# optimisation



