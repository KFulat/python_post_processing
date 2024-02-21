"""Module for plotting basic fourier spectra."""

import numpy as np
import matplotlib
from scipy.ndimage import gaussian_filter, uniform_filter
import scipy.stats as stats
from cycler import cycler

from visualization_package.plot import FigPlot, Plot2D, Plot1D

from shock_visualization.tools import (
    format_step_string,
    parse_arguments_fourier,
    simbox_area,
    curl2D3V_mid,
    curl2D3V_back,
)
from shock_visualization.distrfit import PowerLawFit

from shock_visualization.quantities_to_plot.velocity_linear import (
    vel_linear_dict
)

from shock_visualization.constants import LSI, RES, N0, V0

matplotlib.pyplot.style.use(
    "shock_visualization/basic.mplstyle"
)

matplotlib.rcParams["axes.prop_cycle"] = cycler('color',
    ['black', 'grey', '#EE6677', '#4477AA'])+cycler('linestyle', ['-', '--', '--', '--'])

import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = parse_arguments_fourier()

    for nstep in range(args.start, args.stop + args.step, args.step):
        nstep_string = format_step_string(nstep)
        # x_sh = shock_position_linear(nstep)
        # print(f"shock position: {x_sh}")

        Vx = vel_linear_dict["vel_ion"][0]
        Vy = vel_linear_dict["vel_ion"][1]
        Vz = vel_linear_dict["vel_ion"][2]

        Vx.data_load(nstep)
        Vy.data_load(nstep)
        Vz.data_load(nstep)

        Vx.set_ticks()

        curlVx, curlVy, curlVz = curl2D3V_mid(Vx.data, Vy.data, Vz.data)
        curlV = np.sqrt( curlVx**2 + curlVy**2 + curlVz**2 )
        curlV /= np.abs(V0)

        curlV = gaussian_filter(curlV, sigma=1.0)

        # V = np.sqrt( Vx.data**2 + Vy.data**2 + Vz.data**2 )
        # V = Vx.data

        ticks = [
            0, curlV.shape[1]*RES/LSI,
            0, curlV.shape[0]*RES/LSI
        ]
        levels = (None,None)
        # levels = (0.5e-3,4.5e-3)
        # levels = (0.5e-3,4.5e-3)

        plot= Plot2D(
            # np.ma.log10(curlV),
            curlV,
            loc=(0,0),
            extent=ticks,
            lims=[(0,5.75),(0,5.75)],
            labels=[r"$x/\lambda_{si}$",r"$y/\lambda_{si}$"],
            cmap="turbo",
            levels=levels,
            cbar_label=r"$|\nabla \times \delta v_i|/v_0$",
            cbar_pad="5%",
            cbar_size="5%",
        )

        fig = FigPlot(
            f"../plots/vel/curl/curl_ion_{nstep}.png",
            plot,
            (1,1),
            size=(5,3),
            dpi=150,
            wspace=0.1
        )

        fig.save()