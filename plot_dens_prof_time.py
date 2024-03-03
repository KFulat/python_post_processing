import sys
import numpy as np
import matplotlib
from matplotlib.ticker import MultipleLocator

from visualization_package.plot import Plot2D, FigPlot

from shock_visualization.constants import (
    N0, RES, LSI, PATH_TO_RESULT, STEP_INIT, ION_GYROTIME, MMX, C,
    PATH_TO_PLOT
)
from shock_visualization.quantities import Density
from shock_visualization.tools import (
    load_data_from_h5file, load_Npx, linear_fit
)

matplotlib.pyplot.style.use(
    "shock_visualization/basic.mplstyle"
)
matplotlib.rcParams["image.aspect"] = 'auto'

def calculate_shock_velocity(steps, x_sh):
    v, err, b = linear_fit(steps, x_sh)
    return v

if __name__ == "__main__":
    if len(sys.argv[1:]) == 3:
        start = int(sys.argv[1])
        stop  = int(sys.argv[2])
        step  = int(sys.argv[3])

    x_sh = load_data_from_h5file("./shock_location.h5", "x_sh")
    x_sh_steps = load_data_from_h5file("./shock_location.h5", "steps")
    indx1 = int( (start - x_sh_steps[0]) / 1000.0 )
    indx2 = int( (stop+step - x_sh_steps[0]) / 1000.0 )
    x_sh_steps = x_sh_steps[indx1:indx2]
    x_sh = x_sh[indx1:indx2]
    x_sh_steps = ( x_sh_steps - STEP_INIT ) / ION_GYROTIME

    steps = np.arange(start, stop+step, step)

    Ni = Density(PATH_TO_RESULT,"densiresR",N0,1.5,0.0,norm=True)
    Npx = load_Npx(stop, PATH_TO_RESULT)
    n = (stop+step-start) // step
    prof_map = np.zeros((n, MMX*Npx))
    prof_map[:,:] = -1.0

    for i, nstep in enumerate(steps):
        print(f"step={nstep}")
        Ni.data_load(nstep)
        Ni.data_normalize()
        profile = np.average(Ni.data, axis=0)
        prof_map[i,:profile.shape[0]] = profile

    prof_map = np.ma.log10(prof_map)

    xmax = prof_map.shape[1] * RES
    ymax = (stop - STEP_INIT) / ION_GYROTIME
    ymin = (start - STEP_INIT) / ION_GYROTIME
    ticks = [0, xmax / LSI, ymin, ymax]

    v_sh = calculate_shock_velocity(x_sh_steps*ION_GYROTIME, x_sh*LSI)
    print(f"v_sh/c={v_sh/C}")

    plot = Plot2D(
        prof_map,
        loc=(0,0),
        name="log{}".format(r"$(N_i/N_0)$"),
        extent=ticks,
        labels=[r"$x/\lambda_{si}$", r"$t\Omega_i$"],
        lims=[(0, 100), (ymin, ymax)],
        levels=(-0.05, 1.0),
        cmap="turbo",
        cbar_size="4%",
        cbar_pad="2%",
        major_loc=(10.0,1.0),
        minor_loc=(1.0,0.1),
    )

    fig = FigPlot(
        f"{PATH_TO_PLOT}/dens_profile_time.png",
        plot,
        (1,1),
        size=(6,4),
        hspace=0.05,
        dpi=300,
    )

    fig._plot_axes()
    fig.plots[0].cbar.ax.yaxis.set_major_locator(
        MultipleLocator(0.2)
    )
    fig._savefig()
    fig.close()

