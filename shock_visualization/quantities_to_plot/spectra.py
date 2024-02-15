from shock_visualization.constants import (
    PATH_TO_RESULT, PATH_TO_PLOT, MI, ME, C,
)
from shock_visualization.quantities import PlotParams1D, Spectrum

Ek_ele = Spectrum(PATH_TO_RESULT, "elec", ME*C**2)
Ek_ion = Spectrum(PATH_TO_RESULT, "ions", MI*C**2)

ele_plot_params = PlotParams1D(
    colspan = 1,
    fig_name = "step_ele",
    plot_path = "../ekin",
    plot_name = None,
    limits = [(1e-8, 1e-1), (1e-6, 1e+1)],
    labels = [r"$\gamma-1$", r"$(\gamma-1)N_e(\gamma-1)/N_0$"],
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = (None, None),
    minor_loc = (None, None),
    legend = True,
    legend_position = "upper left",
)

ion_plot_params = PlotParams1D(
    colspan = 1,
    fig_name = "step_ion",
    plot_path = "../ekin",
    plot_name = None,
    limits = [(1e-6, 1e-4), (1e-6, 1e+1)],
    labels = [r"$\gamma-1$", r"$(\gamma-1)N_i(\gamma-1)/N_0$"],
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = (None, None),
    minor_loc = (None, None),
    legend = True,
    legend_position = "upper left",
)

Ek_ele.plot_params = ele_plot_params
Ek_ion.plot_params = ion_plot_params

# Create a dictionary containing all maps
spectra_dict = {
    "Ek_ele": Ek_ele,
    "Ek_ion": Ek_ion,
}
