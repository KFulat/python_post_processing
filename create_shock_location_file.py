import sys
import numpy as np

from shock_visualization.constants import N0, RES, LSI, PATH_TO_RESULT
from shock_visualization.quantities import Density
from shock_visualization.tools import save_data_to_h5file

def shock_location_from_dens(nstep, dens):
    dens.data_load(nstep)
    profile = np.average(dens.data, axis=0)
    index = np.where(profile > 5*N0)[0][-1]
    x_sh = index * RES / LSI
    return x_sh

if __name__ == "__main__":
    if len(sys.argv[1:]) == 3:
        start = int(sys.argv[1])
        stop  = int(sys.argv[2])
        step  = int(sys.argv[3])

    steps = np.arange(start, stop+step, step)
    shock_location = []

    Ni = Density(PATH_TO_RESULT,"densiresR",1.0,1.5,0.0)

    for nstep in steps:
        print(f"step={nstep}")
        shock_location.append(
            shock_location_from_dens(nstep,Ni)
        )
    
    save_data_to_h5file("./shock_location.h5", steps, "steps")
    save_data_to_h5file("./shock_location.h5", shock_location, "x_sh")
