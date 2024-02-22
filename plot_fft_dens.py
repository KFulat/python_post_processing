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
)
from shock_visualization.distrfit import PowerLawFit
from shock_visualization.quantities_to_plot.fourier_dens import (
    fourier_maps_dict
)
from shock_visualization.quantities_to_plot.fourier_bfield import (
    fourier_bfield_dict
)
from shock_visualization.quantities_to_plot.fourier_vel import (
    fourier_vel_dict
)
from shock_visualization.constants import LSI, RES, N0

matplotlib.pyplot.style.use(
    "shock_visualization/basic.mplstyle"
)

matplotlib.rcParams["axes.prop_cycle"] = cycler('color',
    ['black', 'grey', '#EE6677', '#4477AA'])+cycler('linestyle', ['-', '--', '--', '--'])

quantities_dict = {
    **fourier_maps_dict, **fourier_bfield_dict, **fourier_vel_dict
}

if __name__ == "__main__":
    args = parse_arguments_fourier()

    for nstep in range(args.start, args.stop + args.step, args.step):
        nstep_string = format_step_string(nstep)
        # x_sh = shock_position_linear(nstep)
        # print(f"shock position: {x_sh}")

        if args.xlimits and args.ylimits:
            x1, x2, y1, y2 = simbox_area(
                args.xlimits[0],args.xlimits[1],args.ylimits[0],
                args.ylimits[1],LSI,RES
            )
        else:
            a = 10.0
            x1, x2, y1, y2 = simbox_area(a,a+11.52,0,11.52,LSI,RES)

        quantity = quantities_dict[args.quantities[0]][0]
        quantity.data_load(nstep, x1, x2, y1, y2)
        quantity.data_normalize()
        quantity.set_ticks([x1, x2, y1, y2])

        quantity.compute_ft()

        n = quantity.data_ft.shape[0] # x-size of the FT data array
        m = quantity.data_ft.shape[0] # y-size of the FT data array

        # Calculate x- and y-averaged FT data
        ft_x = np.sum(quantity.data_ft, axis=0)
        ft_y = np.sum(quantity.data_ft, axis=1)

        # FFT of the real data is symetrical around 0 BIN (not 0 point)
        # To obtain power spectrum for absolute values of the wavevectors
        # the negative part is flipped and added to the positive part.
        # Then the negative part is set to zero.
        ft_x[n//2+1:] = ft_x[n//2+1:] + np.flip(ft_x[1:n//2])
        ft_x[1:n//2] = 0.0
        ft_x = ft_x[n//2:]
        ft_y[m//2+1:] = ft_y[m//2+1:] + np.flip(ft_y[1:m//2])
        ft_y[1:m//2] = 0.0
        ft_y = ft_y[m//2:]

        kx = quantity.kx
        ky = quantity.ky

        # Calculate magnitude of the wavevectors and respective FT values
        k2D = np.meshgrid(kx,ky)
        knorm = np.sqrt(k2D[0]**2 + k2D[1]**2)
        # knorm = np.sqrt(k2D[0]**2)
        # knorm = k2D[1]
        knorm = knorm.flatten()
        ft = quantity.data_ft.flatten()

        # The wavevectors bins and values.
        dkx = kx[1]-kx[0] # the distance bewteen two wavevector values
        kbins = np.arange(kx[0]-dkx/2, knorm.max()+dkx, dkx)
        kvals = 0.5*(kbins[1:]+kbins[:-1])

        # Power spectrum for the wavector magnitudes
        power_spectrum, _, _ = stats.binned_statistic(
            knorm, ft, statistic = "sum", bins = kbins
        )

        print(f"Sum FT: {np.sum(ft)}")
        print(f"Sum power spectrum: {np.sum(power_spectrum)}")
        print(f"Sum FTx: {np.sum(ft_x)}")
        print(f"Sum FTy: {np.sum(ft_y)}")

        power_spectrum = power_spectrum[n//2:]
        kvals = kvals[n//2:]

        # print(kvals[22])
        # print(power_spectrum[22])

        # # Fit spectrum
        # indx1 = 3
        # indx2 = 8
        # xdata = kvals[indx1:indx2]
        # ydata = power_spectrum[indx1:indx2]

        # from scipy.optimize import curve_fit
        # def power_func(E, a, b):
        #     E = np.array(E)
        #     f = a * E + np.log(b)
        #     return f
        
        # popt, pcov = curve_fit(power_func, np.log(xdata), np.log(ydata))
        # print(f"Curvefit fitted parameters: {popt}")
        # yfit = popt[1]*kvals**popt[0]
        # yfit = np.ma.masked_greater_equal(yfit, 8*1e-3)
        # yfit = np.ma.masked_less_equal(yfit, 1*1e-6)
        # yfit *= 4

        # limits = [(1e-1,1.1e1),(1e-6,2*1e-2)]
        # limits = [
        #     ( np.log10(min(kvals)), np.log10(max(kvals)) ),
        #     ( np.log10(min(power_spectrum)), np.log10(max(power_spectrum)) )
        # ]
        quantity.power_spectrum.append(power_spectrum)
        quantity.power_spectrum.append(ft_x)
        quantity.power_spectrum.append(ft_y)

        quantity.xpoints.append(kvals)
        quantity.xpoints.append(kx[n//2:])
        quantity.xpoints.append(ky[m//2:])

        quantity.add_fourier1d_plot([(0,0), (0,1), (0,2)])
        quantity.plots1D[1].labels[0] = r"$k_x\lambda_{se}$"
        quantity.plots1D[2].labels[0] = r"$k_y\lambda_{se}$"
        quantity.plots1D[1].labels[1] = (
            quantity.plots1D[1].labels[1][:-2] + "_x)$"
        )
        quantity.plots1D[2].labels[1] = (
            quantity.plots1D[2].labels[1][:-2] + "_y)$"
        )
        # quantity.plots1D[0].lims = limits

        # plot_k = Plot1D(
        #     [power_spectrum, yfit],
        #     kvals,
        #     loc=(0,0),
        #     lims=limits,
        #     labels=[r"$k\lambda_{se}$",
        #             "{}".format(
        #                 r"$P_{n_i}(k)$",
        #             )],
        #     )
        
        # plot_kx = Plot1D(
        #     [ft_x],
        #     kx[n//2:],
        #     loc=(0,1),
        #     lims=limits,
        #     labels=[r"$k_x\lambda_{se}$",
        #             "{}".format(
        #                 r"$P_{n_i}(k_x)$",
        #             )],
        #     )
        
        # plot_ky = Plot1D(
        #     [ft_y],
        #     ky[m//2:],
        #     loc=(0,2),
        #     lims=limits,
        #     labels=[r"$k_y\lambda_{se}$",
        #             "{}".format(
        #                 r"$P_{n_i}(k_y)$",
        #             )],
        #     )

        if args.subdirectory:
            fig_path = (
                f"{quantity.plot_params_ft1d.plot_path}/"
                + f"{args.subdirectory}/"
            )
        else:
            fig_path = f"{quantity.plot_params_ft1d.plot_path}/"
        fig_path = (
            fig_path
            + f"{quantity.plot_params_ft1d.fig_name}_{nstep_string}"
            + f".{args.filetype}"
        )
        
        fig = FigPlot(
            fig_path,
            quantity.plots1D,
            (1,3),
            size=(16,3),
            dpi=300,
            wspace=0.4
        )
    
        for plot in fig.plots:
            plot.axis.set_xscale("log", base=10)
            plot.axis.set_yscale("log", base=10)

        # fig.plots[0].axis.text(
        #     1e0, 8*1e-4, "{:.1f}".format(popt[0])
        # )

        fig.save()

        quantity.data_filter()
        quantity.data = gaussian_filter(quantity.data, 2.0)
        quantity.data_logscale()
        quantity.add_fourier_plot((0,0))
        quantity.add_plot((0,1))

        if args.subdirectory:
            fig_path = (
                f"{quantity.plot_params.plot_path}/"
                + f"{args.subdirectory}/"
            )
        else:
            fig_path = f"{quantity.plot_params.plot_path}/"
        fig_path = (
            fig_path
            + f"{quantity.plot_params.fig_name}_{nstep_string}"
            + f".{args.filetype}"
        )

        fig = FigPlot(
            fig_path,
            [quantity.plot, quantity.plot_ft],
            (1,4),
            wspace=0.1,
            size=(13,5),
            dpi=200,
        )

        fig.save()