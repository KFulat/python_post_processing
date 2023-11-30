from shock_visualization.constants import (
    PATH_TO_RESULT, PATH_TO_PLOT, N0, b0, e0, B0, E0
)
from shock_visualization.quantities import Density, Fourier, PlotParams

Ni = Density(PATH_TO_RESULT, "densiresR", N0, 0.0, 0.0)
Ni_ft = Fourier(Ni, "fourier", N0, 0.0, N0)

ft_plot_params = PlotParams(
    colspan = 3,
    fig_name = Ni_ft.data_name,
    plot_path = f"{PATH_TO_PLOT}/dens/{Ni_ft.data_name}",
    plot_name = "a) {}".format(r"$N_i/N_0$"),
    limits = [(None,None), (None,None)],
        # limits = [(Ni_ft.ticks[0], Ni_ft.ticks[1]), (Ni_ft.ticks[2], Ni_ft.ticks[3])],
    labels = [r"$k_x\lambda_{se}$", r"$k_y\lambda_{se}$"],
    cmap = "turbo",
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = (10.0,2.0),
    minor_loc = (1.0,1.0),
    levels = (1.0-0.05, 1.0+0.05),
    cbar_label = "",
)

Ni_ft.plot_params = ft_plot_params

# Create a dictionary containing all maps
fourier_maps_dict = {
    "dens_ft": [Ni_ft],
}