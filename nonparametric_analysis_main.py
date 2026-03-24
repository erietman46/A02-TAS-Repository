import numpy as np
import Datasetcode as ds


#DEFINE VEHICLE DYNAMICS: POSITION, VELOCITY, ACCELERATION

def controlled_element_frf(w, vehicle_type="position", Kc=1.0):
    s = 1j * w

    if vehicle_type.lower() == "position":
        Hc = Kc * np.ones_like(s, dtype=complex)
    elif vehicle_type.lower() == "velocity":
        Hc = Kc / s
    elif vehicle_type.lower() == "acceleration":
        Hc = Kc / (s ** 2)
    else:
        raise ValueError("vehicle_type must be 'position', 'velocity', or 'acceleration'")

    return Hc


#DEFINE OPEN LOOP TRANSFER FUNCTIONS FOR BOTH FIXED BASED AND MOTION BASED 

