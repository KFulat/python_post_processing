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

from shock_visualization.quantities_to_plot.curr_linear import (
    curr_linear_dict
)
from shock_visualization.quantities_to_plot.velocity_linear import (
    vel_linear_dict
)
from shock_visualization.quantities_to_plot.maps_linear import (
    maps_linear_dict
)
from shock_visualization.constants import LSI, RES, N0, V0

matplotlib.pyplot.style.use(
    "shock_visualization/basic.mplstyle"
)

matplotlib.rcParams["axes.prop_cycle"] = cycler('color',
    ['black', 'grey', '#EE6677', '#4477AA'])+cycler('linestyle', ['-', '--', '--', '--'])

quantities_dict = {**curr_linear_dict}

import matplotlib.pyplot as plt

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

        Jx = maps_linear_dict["bfield"][0]
        Jy = maps_linear_dict["bfield"][1]
        Jz = maps_linear_dict["bfield"][2]
        # Jx = curr_linear_dict["curr"][0]
        # Jy = curr_linear_dict["curr"][1]
        # Jz = curr_linear_dict["curr"][2]
        Jx.data_load(nstep)
        Jx.set_ticks()
        # Jx.data_extract()
        # Jx.data_normalize()
        Jy.data_load(nstep)
        # Jy.data_extract()
        # Jy.data_normalize()
        Jz.data_load(nstep)
        # Jz.data_extract()
        # Jz.data_normalize()

        # J = np.sqrt( Jx.data**2 + Jy.data**2 + Jz.data**2 )
        # J = np.sqrt( Jx.data**2 )
        # J = np.abs( Jx.data )
        J = Jz.data

        data_2d = np.array(J)
        # data_2d -= np.mean(data_2d)
        # data_2d /= V0*N0
        ft = np.fft.fft2(data_2d)
        ft = np.fft.fftshift(ft)
        n = ft.shape[0] # x-size of the FT data array
        m = ft.shape[1] # y-size of the FT data array

        # k-space in x and y direction
        kx = np.fft.fftfreq(data_2d.shape[1], 1/5.0)*2*np.pi
        ky = np.fft.fftfreq(data_2d.shape[0], 1/5.0)*2*np.pi

        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        ft_extent = [np.min(kx), np.max(kx), np.min(ky), np.max(ky)]

        def filter_wavenumbers(ft_array,kx,ky,k1,k2):
            # ft[0,:] = 0.0
            # ft[:,0] = 0.0
            # ft[1:n//2, 1:m//2] = 0.0
            # ft[n//2+1:, 1:m//2] = 0.0
            # ft[1:n//2, m//2+1:] = 0.0
            # ft[n//2+1:, m//2+1:] = 0.0
            # ft[n//2,:] = 0.0
            # ft[:,m//2] = 0.0
            ft = np.array(ft_array)
            a = np.array( kx[kx>=k1] )
            a = np.array( a[a<=k2] )
            print(f"wavenumber range: {a}")
            print(f"wavelength range: {2*np.pi/a}")
            k_min = np.min( [np.min(np.abs(kx)), np.min(np.abs(ky))] )
            k_max = np.max( [np.max(np.abs(kx)), np.max(np.abs(ky))] )
            if ( k1 > k_min ):
                indx_x = np.where(np.abs(kx)<k1)
                indx_y = np.where(np.abs(ky)<k1)
                ix1 = np.min(indx_x)
                ix2 = np.max(indx_x)
                iy1 = np.min(indx_y)
                iy2 = np.max(indx_y)
                ft[iy1:iy2+1,ix1:ix2+1] = 0.0
            if ( k2 < k_max ):
                ix = np.min( np.where( kx > k2 ) )
                iy = np.min( np.where( ky > k2 ) )
                ft[iy:, ix:] = 0.0
                ix = np.max( np.where( kx < -k2 ) )
                iy = np.max( np.where( ky < -k2 ) )
                ft[:iy, :ix] = 0.0
            return ft

        print(kx[kx>=0.0])
        
        ft1 = filter_wavenumbers(ft,kx,ky,0.1,0.2)
        ft2 = filter_wavenumbers(ft,kx,ky,0.2,0.3)
        ft3 = filter_wavenumbers(ft,kx,ky,0.9,1.0)

        a = [0.02,0.02,0.02]
        plots = []
        for i, ft in enumerate([ft1,ft2,ft3]):
            ft = np.fft.ifftshift(ft)
            ift = np.fft.ifft2(ft)
            # ift = np.abs(ift)
            ift = ift.real
            mean = np.mean(ift)
            std = np.std(ift)
            # print("sum", np.sum( ((ift-mean)/20.0)**2/ift.size ))
            # print(mean)
            # print(std/20.0*100)
            levels = (mean-2*std,mean+2*std)
            levels = (-a[i],a[i])
            levels = (None,None)

            ift = gaussian_filter(ift, 0.0)

            plot= Plot2D(
                ift,
                loc=(0,i),
                extent=Jx.ticks,
                lims=[(0,5.76),(0,5.76)],
                labels=[r"$x/\lambda_{si}$",r"$y/\lambda_{si}$"],
                cmap="RdBu_r",
                levels=levels,
                cbar_label=r"$\delta N_i/N_0$",
                cbar_pad="5%",
                cbar_size="5%",
            )

            plots.append(plot)

        fig = FigPlot(
            f"../plots/vel/fourier/fourier_ion_{nstep}_filter.png",
            plots,
            (1,3),
            size=(16,3),
            dpi=150,
            wspace=0.4
        )
        for plot, scale in zip(fig.plots,[0.1,0.2,0.3]):
            plot.axis.set_title("{}={}".format(r"$k\lambda_{se}$",scale))

        fig.save()