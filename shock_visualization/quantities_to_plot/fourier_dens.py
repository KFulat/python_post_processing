from shock_visualization.constants import (
    PATH_TO_RESULT, PATH_TO_PLOT, N0, b0, e0, B0, E0
)
from shock_visualization.quantities import Density, Fourier, PlotParams
from shock_visualization.mycolormaps import cmap_ft

Ni = Density(PATH_TO_RESULT, "densiresR", N0, 0.0, 0.0)
Ni_ft = Fourier(Ni, "fourier", N0, 0, 0.0, 6)

plot_params = PlotParams(
    colspan = 3,
    fig_name = Ni_ft.data_name,
    plot_path = f"{PATH_TO_PLOT}/dens/{Ni_ft.data_name}",
    plot_name = "a) {}".format(r"$\delta N_i/N_0$"),
    limits = [(None,None), (None,None)],
    labels = [r"$x/\lambda_{si}$", r"$y/\lambda_{si}$"],
    cmap = "turbo",
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = (10.0,2.0),
    minor_loc = (1.0,1.0),
    # levels = (None,None),
    levels = (-0.1, 0.1),
    cbar_label = "",
    cbar_size = "2%",
    cbar_pad = "2%",
)

ft_plot_params = PlotParams(
    colspan = 1,
    fig_name = Ni_ft.data_name,
    plot_path = f"{PATH_TO_PLOT}/dens/{Ni_ft.data_name}",
    plot_name = None,
    # limits = [(None,None), (None,None)],
    limits = [(None,10), (-10,10)],
    labels = [r"$k_x\lambda_{se}$", r"$k_y\lambda_{se}$"],
    cmap = cmap_ft,
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = (None, None),
    minor_loc = (None, None),
    levels = (-40, -38),
    cbar_label = r"$dP(\delta N_i/N_0)/dk$",
    cbar_size = "10%",
    cbar_pad = "6%",
)

Ni_ft.plot_params = plot_params
Ni_ft.plot_params_ft = ft_plot_params

# Create a dictionary containing all maps
fourier_maps_dict = {
    "dens_ft": [Ni_ft],
}