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
from shock_visualization.constants import LSI, RES, N0

matplotlib.pyplot.style.use(
    "shock_visualization/basic.mplstyle"
)

matplotlib.rcParams["axes.prop_cycle"] = cycler('color',
    ['black', 'grey', '#EE6677', '#4477AA'])+cycler('linestyle', ['-', '--', '--', '--'])

quantities_dict = {**fourier_maps_dict, **fourier_bfield_dict}

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

        quantity.add_plot((0,1))

        quantity.compute_ft()
        quantity.data_filter()
        quantity.add_fourier_plot((0,0))

        n = quantity.data_ft.shape[0] # x-size of the FT data array

        # Calculate y-averaged FT data
        ft_x = np.sum(quantity.data_ft, axis=0)

        # FFT of the real data is symetrical around 0 BIN (not 0 point)
        # To obtain power spectrum for absolute values of the wavevectors
        # the negative part is flipped and added to the positive part.
        # Then the negative part is set to zero.
        ft_x[n//2+1:] = ft_x[n//2+1:] + np.flip(ft_x[1:n//2])
        ft_x[1:n//2] = 0.0

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

        power_spectrum = power_spectrum[n//2:]
        kvals = kvals[n//2:]

        # Fit spectrum
        indx1 = 5
        indx2 = 8
        xdata = kvals[indx1:indx2]
        ydata = power_spectrum[indx1:indx2]

        from scipy.optimize import curve_fit
        def power_func(E, a, b):
            E = np.array(E)
            f = a * E + np.log(b)
            return f
        
        popt, pcov = curve_fit(power_func, np.log(xdata), np.log(ydata))
        print(f"Curvefit fitted parameters: {popt}")
        yfit = popt[1]*kvals**popt[0]
        yfit = np.ma.masked_greater_equal(yfit, 8*1e-3)
        yfit = np.ma.masked_less_equal(yfit, 1*1e-6)
        yfit *= 4

        plotx = Plot1D(
            # [ft_x, power_spectrum[:-1]],
            [power_spectrum, yfit],
            # [ft_x, yfit],
            kvals,
            # kvals[:-1],
            loc=(0,0),
            # lims=[(1e-1,2*1e0),(8*1e-6,1e-1)],
            lims=[(1e-1,10*1e0),(1e-6,2*1e-2)],
            # lims=[(0,5),(1e-6,1e-2)],
            # lims=[(None,None),(None,None)],
            labels=[r"$k\lambda_{se}$",
                    "{}".format(
                        r"$P_{n_i}(k)$",
                    )],
            # major_loc=(0.5,None),
            )
        
        fig = FigPlot(
            f"../plots/dens/fourier/fourier_{nstep}_1D.png",
            [plotx],
            (1,1),
            size=(5,4),
            dpi=300,
            hspace=0.4
        )
    
        fig.plots[0].axis.set_xscale("log", base=10)
        fig.plots[0].axis.set_yscale("log", base=10)

        fig.plots[0].axis.text(
            1e0, 8*1e-4, "{:.1f}".format(popt[0])
        )

        fig.save()

        exit()

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

        exit()

        for comp in N:
            # remove mean values and external gradients
            comp.data -= np.mean(comp.data)
            data_filtered = gaussian_filter(comp.data , 30)
            comp.data -= data_filtered

            # calculate fourier spectrum
            ft, ticks, kx, ky = spatial_ft(comp.data, 5, True, False, False)
            N_ft.append(ft)

        # Normalization and energy density
        norm = 1.0 / N0
        N_ft = [comp*norm**2 for comp in N_ft]

        for i, ft in enumerate(N_ft):
            # chose specific region of the tranform
            n = ft.shape[0]
            m = ft.shape[1]
            N_ft[i] = 2.0 * ft[:, m//2+1:] # multiplied by two to maintain norm
                    
            # smooth the power spectra
            N_ft[i] = uniform_filter(N_ft[i], 3)

        # Change ticks
        kx_abs = kx[kx>0.0]
        ky_abs = ky[ky>0.0]
        ticks = [np.min(kx_abs), np.max(kx_abs), np.min(ky), np.max(ky)]

        print(ticks)

        x1 = ticks[0]
        x2 = 8.0
        y1 = -10.0
        y2 = 10.0
        ticks_field = [xL*RES/LSI, xR*RES/LSI, yB*RES/LSI, yU*RES/LSI]

        for i, (ft, comp) in enumerate(zip(N_ft, ["i", "e"])):
            ft_x = np.sum(ft, axis=0)
            ft_y = np.sum(ft, axis=1)

            indx = kx_abs >= x1
            kx_abs_tr = kx_abs[indx]
            ft_x = ft_x[indx]
            indx = kx_abs_tr <= x2
            kx_abs_tr = kx_abs_tr[indx]
            ft_x = ft_x[indx]

            indx = ky >= y1
            ky_tr = ky[indx]
            ft_y = ft_y[indx]
            indx = ky_tr <= y2
            ky_tr = ky_tr[indx]
            ft_y = ft_y[indx]

            kx_mean = np.sum(kx_abs_tr*ft_x) / np.sum(ft_x)
            ky_mean = np.sum(ky_tr*ft_y) / np.sum(ft_y)
            k_mean = np.sqrt( kx_mean**2 + ky_mean**2 )

            kx_max = kx_abs_tr[np.argmax(ft_x)]
            ky_max = ky_tr[np.argmax(ft_y)]
            k_mean_max = np.sqrt( kx_max**2 + ky_max**2 )

            limsy = (-5,-2)

            plotx = Plot1D(
                    np.ma.log10(ft_x),
                    kx_abs_tr,
                    loc=(0,0),
                    lims=[(x1,x2),limsy],
                    labels=[r"$k_x\lambda_{se}$",
                            "log({})".format(
                                r"$dP(N_{})/dk_x$".format(comp),
                            )],
                    major_loc=(2.0,None),
                    )

            ploty = Plot1D(
                    np.ma.log10(ft_y),
                    ky_tr,
                    loc=(1,0),
                    colspan=2,
                    lims=[(y1,y2),limsy],
                    labels=[r"$k_y\lambda_{se}$",
                            "log({})".format(
                                r"$dP(N_{})/dk_y$".format(comp),
                            )],
                    major_loc=(2.0,None),
                    )

            fig = FigPlot(f"../dens/fourier/fourier_{nstep}_1D_"
                          + f"{int(ticks_field[0])}_{int(ticks_field[1])}_N{comp}.png",
                        [plotx, ploty], (2,2),
                        size=(1.8*1*3.36, 1.2*2*2.725),
                        dpi=300, hspace=0.4)
            
            dx = 0.2
            dy = 0.5
            yloc = limsy[1] - dy
            fig.plots[0].axis.axvline(kx_mean, ls="--", c="gray")
            fig.plots[0].axis.text(kx_mean+dx, yloc,
                                "{}={:.2f}".format(r"$\overline{k_x}\lambda_{se}$",kx_mean), 
                                )
            fig.plots[1].axis.axvline(ky_mean, ls="--", c="gray")
            fig.plots[1].axis.text(ky_mean+dx, yloc,
                                "{}={:.2f}".format(r"$\overline{k_y}\lambda_{se}$",ky_mean),
                                )
            yloc = limsy[0] + dy
            fig.plots[1].axis.text(y1, yloc - 4*dy,
                                "The mean wavenumber: {}={:.2f}".format(r"$\overline{k}\lambda_{se}$",k_mean))
            fig.plots[0].axis.axvline(kx_max, ls="--", c="gray")
            fig.plots[0].axis.text(kx_max+dx, yloc,
                                "{}={:.2f}".format(r"$k_{x,peak}\lambda_{se}$",kx_max),
                                )
            fig.plots[1].axis.axvline(ky_max, ls="--", c="gray")
            fig.plots[1].axis.text(ky_max+dx, yloc,
                                "{}={:.2f}".format(r"$k_{y,peak}\lambda_{se}$",ky_max),
                                )
            fig.plots[1].axis.text(y1, yloc - 5*dy,
                                "The peak wavenumber: {}={:.2f}".format(r"$k_{peak}\lambda_{se}$",k_mean_max))
            fig.save()

        for i, ft in enumerate(N_ft):
            # logscale
            N_ft[i] = np.log10(N_ft[i])

        for comp in N:
            comp.data_normalize()
            comp.data_logscale()

        # PLOT
        ticks_field = [xL*RES/LSI, xR*RES/LSI, yB*RES/LSI, yU*RES/LSI]
        levels = (-7, -4.5)
        levels_field = (-1.5, -0.5)
        # levels_field = (-0.03, 0.03)
        lims = [(ticks[0],x2),(y1,y2)]

        cbar_size = "5.0%"
        cbar_pad = "3.5%"

        plot4 = Plot2D(N[1].data, loc=(0, 1), name="b) log{}".format(r"$N_e$"),
                extent=ticks_field,
                levels=levels_field,
                cmap="turbo",
                # lims=lims,
                rowspan=2,
                colspan=1,
                hide_xlabels=False,
                labels=[r"$x/\lambda_{si}$", r"$y/\lambda_{si}$"],
                cbar_size=cbar_size,
                cbar_pad=cbar_pad,
                )
        plot3 = Plot2D(N[0].data, loc=(0, 0), name="a) log{}".format(r"$N_i$"),
                sharex = plot4,
                extent=ticks_field,
                levels=levels_field,
                cmap="turbo",
                # lims=lims,
                rowspan=2,
                colspan=1,
                hide_xlabels=False,
                labels=[r"$x/\lambda_{si}$", r"$y/\lambda_{si}$"],
                cbar_size=cbar_size,
                cbar_pad=cbar_pad,
                )

        plot2 = Plot2D(N_ft[1], loc=(2, 1), name="d) log{}".format(r"$(\text{d}P(N_{e})/\text{d}k)$"),
                # sharex = plot3,
                extent=ticks,
                levels=levels,
                cmap=cmap_ft_turbo,
                lims=lims,
                rowspan=3,
                colspan=1,
                hide_xlabels=False,
                labels=[r"$k_x\lambda_{se}$", r"$k_y\lambda_{se}$"],
                cbar_size=cbar_size,
                cbar_pad=cbar_pad,
                )
        
        plot1 = Plot2D(N_ft[0], loc=(2, 0), name="c) log{}".format(r"$(\text{d}P(N_{i})/\text{d}k)$"),
                sharex = plot2,
                extent=ticks,
                levels=levels,
                cmap=cmap_ft_turbo,
                lims=lims,
                rowspan=3,
                colspan=1,
                hide_xlabels=False,
                labels=[r"$k_x\lambda_{se}$", r"$k_y\lambda_{se}$"],
                cbar_size=cbar_size,
                cbar_pad=cbar_pad,
                )


        fig = FigPlot(f"../dens/fourier_turbulence/fourier_{nstep}_"
                      + f"{int(ticks_field[0])}_{int(ticks_field[1])}.png",
                      [plot1, plot2, plot3, plot4],
                      (5, 6),
                      size=(2.2*6*3.36/1.5, 1.9*5*2.725/1.5),
                      hspace=0.01, wspace=0.6, dpi=200)

        fig._plot_axes()

        # from matplotlib.ticker import MultipleLocator
        # fig.plots[0].cbar.ax.yaxis.set_major_locator(MultipleLocator(.25))

        # string = "{}{:.2e}{}".format(r"$U_{B}=$", U_B, r"$\cdot U_{B0}$")
        # fig.plots[3].axis.text(xL*RES/LSI, 12.5, string, fontsize=32)

        fig._savefig()
        fig.close()