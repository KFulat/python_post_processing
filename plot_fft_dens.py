"""Module for plotting basic shock quantities."""

import numpy as np
import matplotlib
from scipy import fft, interpolate
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter

from visualization_package.plot import FigPlot, Plot2D

from shock_visualization.tools import (
    format_step_string,
    parse_arguments,
)
from shock_visualization.quantities_to_plot.fourier_dens import (
    fourier_maps_dict
)
from shock_visualization.constants import LSI, RES, PATH_TO_RESULT, N0, PATH_TO_PLOT
from shock_visualization.quantities import Density, Fourier, PlotParams

matplotlib.pyplot.style.use(
    "shock_visualization/basic.mplstyle"
)

quantities_dict = {**fourier_maps_dict}

if __name__ == "__main__":
    args = parse_arguments()

    for nstep in range(args.start, args.stop + args.step, args.step):
        nstep_string = format_step_string(nstep)
        # x_sh = shock_position_linear(nstep)
        # print(f"shock position: {x_sh}")

        def simbox_area(x1,x2,y1,y2,input_unit,res_factor):
            """Are of the simulation box converted from input to resized unit"""
            x1 = int( x1 * input_unit/res_factor)
            x2 = int( x2 * input_unit/res_factor)
            x1 = int( y1 * input_unit/res_factor)
            y2 = int( y2 * input_unit/res_factor)
            return x1, x2, y1, y2

        a = 0
        x1, x2, y1, y2 = simbox_area(a,a+21.52,0,11.52,LSI,RES)

        Ni = Density(PATH_TO_RESULT, "densiresR", N0, 0.0, 0.0)
        Ni.data_load(nstep)

        Ni_ft = quantities_dict[args.quantities[0]][0]
        Ni_ft.data_load(nstep)
        Ni_ft.data_normalize()
        Ni_ft.data = Ni_ft.data[x1:x2, y1:y2]
        Ni_ft.set_ticks([x1, x2, y1, y2])

        Ni_ft.add_plot((0,1))

        Ni_ft.compute_ft()
        Ni_ft.add_fourier_plot((0,0))

        fig = FigPlot(
            f"../plots/dens/fourier/fourier_{nstep}_"
            + f"{int(x1)}_{int(x2)}.png",
            [Ni_ft.plot, Ni_ft.plot_ft],
            (1,4),
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