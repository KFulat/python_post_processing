import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import curve_fit
from scipy.special import kn, gamma

from shock_visualization.constants import C

class DistrFitBase(ABC):
    def __init__(self, xdata, ydata, xmin, xmax, bin_min, bin_delta):
        self.xdata = xdata
        self.ydata = ydata
        self.xmin = xmin
        self.xmax = xmax
        self.bin_min = bin_min
        self.bin_delta = bin_delta
        self.yfit = None
        self.popt = None
    
    def select_data_range(self):
        imin = int( np.log10(self.xmin/self.bin_min) / self.bin_delta )
        imax = int( np.log10(self.xmax/self.bin_min) / self.bin_delta )

        xdata_to_fit = np.array( self.xdata[imin:imax] )
        ydata_to_fit = np.array( self.ydata[imin:imax] )
        
        return xdata_to_fit, ydata_to_fit
    
    @staticmethod
    def print_fitted_params(popt, perr):
        print("")
        print("-------------------------------------------------------")
        print(f"Curvefit fitted parameters: {popt}")
        print(f"Curvefit err fitted parameters: {perr}")
        print("-------------------------------------------------------")

    @abstractmethod
    def fit_function(self):
        pass
    
    @abstractmethod
    def fit_with_curvefit(self):
        pass

class MaxwellFit(DistrFitBase):
    def __init__(self, xdata, ydata, xmin, xmax, bin_min, bin_delta):
        super().__init__(xdata, ydata, xmin, xmax, bin_min, bin_delta)

    def fit_function(self, E, a, b):
        # E - kinetic energy, a = 2/log10(e)/sqrt(PI), b = 1/kT
        E = np.array(E)
        f = a*E**(3/2)*np.exp(-b*E)
        return f

    def fit_with_curvefit(self):
        x, y = self.select_data_range()
        popt, pcov = curve_fit(self.fit_function, x, y)
        perr = np.sqrt(np.diag(pcov))
        self.print_fitted_params(popt, perr)

        yfit = self.fit_function(self.xdata, popt[0], popt[1])

        self.yfit = yfit
        self.popt = popt

class JuttnerFit(MaxwellFit):
    def __init__(self, xdata, ydata, xmin, xmax, bin_min, bin_delta):
        super().__init__(xdata, ydata, xmin, xmax, bin_min, bin_delta)

    def fit_function(self, E, t, A):
        # E - kinetic energy, a = 1/log10(e), b = 1/theta = mc^2/kT
        E = np.array(E)
        f = (
            A * np.exp(-1.0/t)/(t*kn(2,1.0/t)) * np.exp(-E/t) * (1+E)
            * np.sqrt((E+1.0)**2-1.0) * E
        )
        return f
    
class KappaFit(DistrFitBase):
    def __init__(self, xdata, ydata, xmin, xmax, bin_min, bin_delta):
        super().__init__(xdata, ydata, xmin, xmax, bin_min, bin_delta)

    def fit_function(self, E, k, theta, A):
        # E - kinetic energy
        E = np.array(E)
        B = 2*C**2
        f = ( A * (k*theta**2)**(-3/2) * gamma(k+1.0)/gamma(k-0.5)
            * ( 1.0 + B*E/(k*theta**2) )**(-(k+1.0)) * E**(3/2)
        ) 
        return f
    
    def fit_with_curvefit(self):
        x, y = self.select_data_range()
        popt, pcov = curve_fit(self.fit_function, x, y)
        perr = np.sqrt(np.diag(pcov))
        self.print_fitted_params(popt, perr)

        yfit = self.fit_function(self.xdata, popt[0], popt[1], popt[2])

        self.yfit = yfit
        self.popt = popt

class PowerLawFit(DistrFitBase):
    def __init__(self, xdata, ydata, xmin, xmax, bin_min, bin_delta):
        super().__init__(xdata, ydata, xmin, xmax, bin_min, bin_delta)

    def fit_function(self, E, alpha, A):
        E = np.array(E)
        f = alpha*E + np.log(A)
        return f
    
    def fit_with_curvefit(self):
        x, y = self.select_data_range()
        popt, pcov = curve_fit(self.fit_function, np.log(x), np.log(y))
        perr = np.sqrt(np.diag(pcov))
        self.print_fitted_params(popt, perr)

        yfit = popt[1] * self.xdata**popt[0]
        yfit = np.ma.masked_greater_equal(yfit, 1e-2)

        self.yfit = yfit
        self.popt = popt