import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from matplotlib.colors import Colormap
from xmlrpc.client import boolean
from scipy.ndimage import gaussian_filter, uniform_filter
from copy import deepcopy

from visualization_package.plot import Plot1D, Plot2D
from shock_visualization.tools import (
    load_movHR, load_ekin, field_log_new, load_phase, load_Npx,
    load_mom_distr, spatial_ft, curl2D3V_mid, div2D3V_back
)
from shock_visualization.constants import RES, LSI, RES_PHASE, MMX, MMY, C


class QuantityBase(ABC):
    def __init__(self, data_path, data_name, data_norm_const, filter_level,
                 log=False, norm=False):
        self.data = None
        self.data_path = data_path
        self.data_name = data_name
        self.data_norm_const = data_norm_const
        self.filter_level = filter_level
        self.plot_params = None
        self.ticks = None
        self.plot = None
        self.log = log
        self.norm = norm

    @abstractmethod
    def data_load(self):
        pass

    @abstractmethod
    def data_logscale(self):
        pass

    @abstractmethod
    def data_normalize(self):
        pass

    @abstractmethod
    def add_plot(self):
        pass

@dataclass
class PlotParams:
    colspan: int
    fig_name: str
    plot_path: str
    plot_name: list
    limits: list
    labels: list
    cmap: Colormap
    hide_xlabels: boolean
    hide_ylabels: boolean
    major_loc: tuple
    minor_loc: tuple
    levels: tuple
    cbar_label: str
    cbar_size: str
    cbar_pad: str

@dataclass
class PlotParams1D:
    colspan: int
    fig_name: str
    plot_path: str
    plot_name: list
    limits: list
    labels: list
    hide_xlabels: boolean
    hide_ylabels: boolean
    major_loc: tuple
    minor_loc: tuple
    legend: boolean
    legend_position: str

class Spectrum(QuantityBase):
    def __init__(self, data_path, data_name, data_norm_const):
        super().__init__(data_path, data_name, data_norm_const, 1.0, False, True)
        self.data = []
        self.xpoints = None
        self.delta = 1
        self.bin_min = 1

    def data_load(self, nstep, xL, xR):
        self.data = []
        data = load_ekin(nstep, self.data_name, self.data_path)
        # attrs = group[self.data_name]['attrs']
        # bin_min = attrs['bmin']
        # bin_max = attrs['bmax']
        # bin_num = attrs['bnum']
        bin_min = 0.00001
        bin_max = 100.0
        bin_num = 500
        bin_delta = np.log10(bin_max / bin_min) / bin_num
        print(bin_min, bin_max, bin_num, bin_delta)
        # data = group[self.data_name][self.data_name].T
        data = data.T
        # x_min = attrs['xmin']
        # x_max = attrs['xmax']
        # x_num = attrs['xnum']
        x_min = 0
        x_max = 960
        x_num = 10
        a = int(xL*LSI / x_max * x_num)
        b = int(xR*LSI / x_max * x_num)
        data = data[a:b,:,:]
        data = data.sum(axis=(0,1))
        data = data / np.sum(data) / bin_delta
        xpoints = bin_min*(10.0**(bin_delta*np.arange(bin_num)))
        self.delta = bin_delta
        self.xpoints = xpoints
        self.data = data
        self.bin_min = bin_min

    def data_logscale(self):
        pass

    def data_normalize(self):
        if self.norm: self.xpoints = [x / self.data_norm_const for x in self.xpoints]

    def add_plot(self, loc):
        plot = Plot1D(
            self.data,
            self.xpoints,
            loc=loc,
            plot_names=self.plot_params.plot_name,
            rowspan=1,
            colspan=self.plot_params.colspan,
            labels=self.plot_params.labels,
            lims=self.plot_params.limits,
            hide_xlabels=self.plot_params.hide_xlabels,
            legend=self.plot_params.legend,
            legend_position=self.plot_params.legend_position,
            major_loc=self.plot_params.major_loc,
            minor_loc=self.plot_params.minor_loc,
        )
        self.plot = plot


class ProfileX(QuantityBase):
    def __init__(
            self, data_path, data_name, data_norm_const, filter_level,
            quantities,
            log=False, norm=False, abs=True,
        ):
        super().__init__(
            data_path, data_name, data_norm_const,
            filter_level, log, norm
        )
        self.data = []
        self.quantities = quantities
        self.xpoints = None
        self.abs = abs
    
    def data_load(self, nstep):
        self.data = []
        for i, quantity in enumerate(self.quantities):
            quantity.data_load(nstep)
            if self.abs: quantity.data = np.abs(quantity.data)
            data = np.average(quantity.data, axis=0)
            data = gaussian_filter(data, sigma=self.filter_level)
            self.data.append(data)
        data_size = min([arr.size for arr in self.data])
        self.data = [arr[:data_size] for arr in self.data]
        xmax = data_size * RES
        self.xpoints = np.linspace(
            0, xmax/LSI, num=data_size
        )

    def data_logscale(self):
        if self.log: self.data = [np.ma.log10(x) for x in self.data]

    def data_normalize(self):
        if self.norm: self.data = [x / self.data_norm_const for x in self.data]

    def data_extract(self):
        pass

    def set_ticks(self):
        self.ticks = [self.xpoints[0], self.xpoints[-1]]
        if self.plot_params.limits[0] == (None, None):
            self.plot_params.limits[0] = (self.ticks[0], self.ticks[1])

    def add_plot(self, loc):
        plot = Plot1D(
            self.data,
            self.xpoints,
            loc=loc,
            plot_names=self.plot_params.plot_name,
            rowspan=1,
            colspan=self.plot_params.colspan,
            labels=self.plot_params.labels,
            lims=self.plot_params.limits,
            hide_xlabels=self.plot_params.hide_xlabels,
            legend=self.plot_params.legend,
            legend_position=self.plot_params.legend_position,
            major_loc=self.plot_params.major_loc,
            minor_loc=self.plot_params.minor_loc,
        )
        self.plot = plot

class Density(QuantityBase):
    def __init__(
            self, data_path, data_name, data_norm_const, filter_level,
            data_extract_const, log=False, norm=False,
            extract=False
    ):
        super().__init__(data_path, data_name, data_norm_const, filter_level, log, norm)
        self.data_extract_const = data_extract_const
        self.extract = extract

    def data_load(self, nstep):
        data = load_movHR(nstep, self.data_name, self.data_path)
        data = data.T[2:-2, 2:-2]
        data = gaussian_filter(data, sigma=self.filter_level)
        self.data = data
    
    def data_logscale(self):
        if self.log: self.data = np.ma.log10(self.data)

    def data_normalize(self):
        if self.norm: self.data /= self.data_norm_const

    def data_extract(self):
        if self.extract: self.data -= self.data_extract_const

    def set_ticks(self):
        xmax = self.data.shape[1] * RES
        ymax = self.data.shape[0] * RES
        self.ticks = [0, xmax / LSI, 0, ymax / LSI]
        if self.plot_params.limits[1] == (None, None):
            self.plot_params.limits[1] = (0, ymax / LSI)

    def add_plot(self, loc):
        plot = Plot2D(
            self.data,
            loc=loc,
            name=self.plot_params.plot_name,
            extent=self.ticks,
            rowspan=1,
            colspan=self.plot_params.colspan,
            labels=self.plot_params.labels,
            lims=self.plot_params.limits,
            levels=self.plot_params.levels,
            hide_xlabels=self.plot_params.hide_xlabels,
            cmap=self.plot_params.cmap,
            cbar_size=self.plot_params.cbar_size,
            cbar_pad=self.plot_params.cbar_pad,
            cbar_extend = "neither",
            cbar_label = self.plot_params.cbar_label,
            major_loc=self.plot_params.major_loc,
            minor_loc=self.plot_params.minor_loc,
        )
        self.plot = plot

class Field(Density):
    def __init__(self, data_path, data_name, data_norm_const, filter_level, data_extract_const, log=False, norm=False, extract=False):
        super().__init__(data_path, data_name, data_norm_const, filter_level, data_extract_const, log, norm, extract)

    def data_logscale(self):
        if self.log: self.data = field_log_new(self.data, 0.01)

class Velocity(Field):
    def __init__(
            self, data_norm_const, filter_level,
            data_extract_const, curr, dens,
            log=False, norm=False, extract=False
    ):
        super().__init__(
            "", "", data_norm_const, filter_level,
            data_extract_const, log, norm, extract
        )
        self.curr = curr
        self.dens = dens

    def data_load(self, nstep):
        self.curr.data_load(nstep)
        self.dens.data_load(nstep)
        data = self.curr.data / self.dens.data
        data = gaussian_filter(data, sigma=self.filter_level)
        self.data = data        

class CurrentTotal(Field):
    def __init__(
            self, data_norm_const, filter_level,
            data_extract_const, curr_ion, curr_ele,
            log=False, norm=False, extract=False
    ):
        super().__init__(
            "", "", data_norm_const, filter_level,
            data_extract_const, log, norm, extract
        )
        self.curr_ion = curr_ion
        self.curr_ele = curr_ele

    def data_load(self, nstep):
        self.curr_ion.data_load(nstep)
        self.curr_ele.data_load(nstep)
        data = self.curr_ion.data - self.curr_ele.data
        data = gaussian_filter(data, sigma=self.filter_level)
        self.data = data

class CurlOfField(Field):
    def __init__(
            self, data_norm_const, filter_level,
            data_extract_const, field_x, field_y, field_z,
            log=False, norm=False, extract=False
    ):
        super().__init__(
            "", "", data_norm_const, filter_level,
            data_extract_const, log, norm, extract
        )
        self.field_x = field_x
        self.field_y = field_y
        self.field_z = field_z

    def data_load(self, nstep):
        self.field_x.data_load(nstep)
        self.field_y.data_load(nstep)
        self.field_z.data_load(nstep)
        curl_x, curl_y, curl_z = curl2D3V_mid(
            self.field_x.data, self.field_y.data, self.field_z.data
        )
        curl = np.sqrt( curl_x**2 + curl_y**2 + curl_z**2 )
        curl = gaussian_filter(curl, sigma=self.filter_level)
        self.data = curl

class Magnitude(Field):
    def __init__(
            self, data_norm_const, filter_level,
            data_extract_const, vx, vy, vz,
            log=False, norm=False, extract=False
    ):
        super().__init__(
            "", "", data_norm_const, filter_level,
            data_extract_const, log, norm, extract
        )
        self.vx = vx
        self.vy = vy
        self.vz = vz

    def data_load(self, nstep):
        self.vx.data_load(nstep)
        self.vy.data_load(nstep)
        self.vz.data_load(nstep)
        self.vx.data_extract()
        self.vy.data_extract()
        self.vz.data_extract()
        self.vx.data_normalize()
        self.vy.data_normalize()
        self.vz.data_normalize()
        v = np.sqrt( 
            self.vx.data**2 + self.vy.data**2 + self.vz.data**2
        )
        v = gaussian_filter(v, sigma=self.filter_level)
        self.data = v

class DivergenceOfField(Field):
    def __init__(
            self, data_norm_const, filter_level,
            data_extract_const, field_x, field_y, field_z,
            log=False, norm=False, extract=False
    ):
        super().__init__(
            "", "", data_norm_const, filter_level,
            data_extract_const, log, norm, extract
        )
        self.field_x = field_x
        self.field_y = field_y
        self.field_z = field_z

    def data_load(self, nstep):
        self.field_x.data_load(nstep)
        self.field_y.data_load(nstep)
        self.field_z.data_load(nstep)
        div = div2D3V_back(
            self.field_x.data, self.field_y.data, self.field_z.data
        )
        div = gaussian_filter(div, sigma=self.filter_level)
        self.data = div

class Phase(Density):
    def __init__(self, data_path, data_name, data_norm_const, filter_level, log=False, norm=False):
        super().__init__(data_path, data_name, data_norm_const, filter_level, log, norm)

    def data_load(self, nstep):
        data = load_phase(nstep, self.data_name, self.data_path)
        data = data.T
        # print(data[:,1360:1400])
        # data = data[:,1380:1390]
        # data[:500, :] = 0.0
        self.data = data

    def set_ticks(self):
        xmax = self.data.shape[1] * RES_PHASE
        yarr = (np.arange(800)-800.0/2)*0.01
        self.ticks = [0, xmax / LSI, yarr[0], yarr[-1]]

class Momentum(Density):
    def __init__(self, data_path, data_name, data_norm_const, filter_level, log=False, norm=False):
        super().__init__(data_path, data_name, data_norm_const, filter_level, log, norm)
    
    def data_load(self, nstep):
        group = load_mom_distr(nstep, self.data_name, self.data_path)
        attrs = group[self.data_name]['attrs']
        self.x_min = float(attrs['xmin_val'])
        self.x_max = float(attrs['xmax_val'])
        self.y_min = float(attrs['ymin_val'])
        self.y_max = float(attrs['ymax_val'])
        data = group[self.data_name][self.data_name]
        self.data = data
    
    def set_ticks(self):
        self.ticks = [self.x_min/C, self.x_max/C, self.y_min/C, self.y_max/C]

class Fourier(QuantityBase):
    def __init__(
            self, quantity, data_name, data_norm_const, filter_level,
            data_extract_const, uniform_filter, log=True, norm=True,
            extract=True
    ):
        super().__init__(
            "", data_name, data_norm_const, filter_level, log, norm
        )
        self.quantity = quantity
        self.data_extract_const = data_extract_const
        self.extract = extract
        self.plot_params_ft = None
        self.plot_params_ft1d = None
        self.uniform_filter = uniform_filter
        self.xpoints = []
        self.power_spectrum = []
        self.plots1D = []

    def data_load(self, nstep, x1,x2,y1,y2):
        self.quantity.data_load(nstep)
        self.data = self.quantity.data
        self.data = self.data[y1:y2, x1:x2]

    def data_filter(self):
        self.data_ft = uniform_filter(self.data_ft, self.uniform_filter)

    def compute_ft(self):
        self.data -= np.mean(self.data)
        if self.filter_level:
            data_filtered = gaussian_filter(self.data ,
                                            self.filter_level)
            self.data -= data_filtered
        ft, ticks, kx, ky = spatial_ft(self.data, 5, normalize=True,
                                       hanning=False, log=False)
        
        print(f"Mean from data: {np.sum(self.data**2)/self.data.size}")
        print(f"Sum from FT: {np.sum(ft)}")
        # chose specific region of the tranform
        # n = ft.shape[0]
        # m = ft.shape[1]

        # ft = 2.0 * ft[:, m//2+1:] # multiplied by two to maintain norm
        # ft = ft[n//2+1, :] + np.flipud(ft[:n//2-1,:])
        self.data_ft = ft
        # kx_abs = kx[kx>0.0]
        # ky_abs = ky[ky>0.0]
        # ticks[0] = np.min(kx_abs)
        # ticks[2] = np.min(ky_abs)
        self.ticks_ft = ticks
        self.kx = kx
        self.ky = ky
        if self.plot_params_ft.limits == [(None, None), (None, None)]:
            self.plot_params_ft.limits[0] = (self.ticks_ft[0], self.ticks_ft[1])
            self.plot_params_ft.limits[1] = (self.ticks_ft[2], self.ticks_ft[3])
    
    def data_logscale(self):
        if self.log: self.data_ft = np.ma.log10(self.data_ft)

    def data_normalize(self):
        if self.norm: self.data /= self.data_norm_const

    def data_extract(self):
        if self.extract: self.data -= self.data_extract_const

    def set_ticks(self, box_area):
        self.ticks = np.array(box_area) * RES/LSI
        if self.plot_params.limits == [(None, None), (None, None)]:
            self.plot_params.limits[0] = (self.ticks[0], self.ticks[1])
            self.plot_params.limits[1] = (self.ticks[2], self.ticks[3])

    def add_plot(self, loc):
        plot = Plot2D(
            self.data,
            loc=loc,
            name=self.plot_params.plot_name,
            extent=self.ticks,
            rowspan=1,
            colspan=self.plot_params.colspan,
            labels=self.plot_params.labels,
            lims=self.plot_params.limits,
            levels=self.plot_params.levels,
            hide_xlabels=self.plot_params.hide_xlabels,
            cmap=self.plot_params.cmap,
            cbar_size=self.plot_params.cbar_size,
            cbar_pad=self.plot_params.cbar_pad,
            cbar_extend = "neither",
            cbar_label = self.plot_params.cbar_label,
            major_loc=self.plot_params.major_loc,
            minor_loc=self.plot_params.minor_loc,
        )
        self.plot = plot

    def add_fourier_plot(self, loc):
        plot = Plot2D(
            self.data_ft,
            loc=loc,
            name=self.plot_params_ft.plot_name,
            extent=self.ticks_ft,
            rowspan=1,
            colspan=self.plot_params_ft.colspan,
            labels=self.plot_params_ft.labels,
            lims=self.plot_params_ft.limits,
            levels=self.plot_params_ft.levels,
            hide_xlabels=self.plot_params_ft.hide_xlabels,
            cmap=self.plot_params_ft.cmap,
            cbar_size=self.plot_params_ft.cbar_size,
            cbar_pad=self.plot_params_ft.cbar_pad,
            cbar_extend = "neither",
            cbar_label = self.plot_params_ft.cbar_label,
            major_loc=self.plot_params_ft.major_loc,
            minor_loc=self.plot_params_ft.minor_loc,
        )
        self.plot_ft = plot

    def add_fourier1d_plot(self, loc):
        for i in range(3):
            plot_params_ft1d = deepcopy(self.plot_params_ft1d)
            plot = Plot1D(
                self.power_spectrum[i],
                self.xpoints[i],
                loc=loc[i],
                plot_names=plot_params_ft1d.plot_name,
                rowspan=1,
                colspan=plot_params_ft1d.colspan,
                labels=plot_params_ft1d.labels,
                lims=plot_params_ft1d.limits,
                hide_xlabels=plot_params_ft1d.hide_xlabels,
                legend=plot_params_ft1d.legend,
                legend_position=plot_params_ft1d.legend_position,
                major_loc=plot_params_ft1d.major_loc,
                minor_loc=plot_params_ft1d.minor_loc,
            )
            self.plots1D.append(plot)