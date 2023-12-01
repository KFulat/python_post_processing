from shock_visualization.constants import (
    PATH_TO_RESULT, PATH_TO_PLOT, b0, B0
)
from shock_visualization.quantities import Field, Fourier, PlotParams
from shock_visualization.mycolormaps import cmap_ft

Bx = Field(PATH_TO_RESULT, "bxres", b0, 0.0, B0[0])
Bx_ft = Fourier(Bx, "fourier", b0, 0, 0.0, 6)

plot_params = PlotParams(
    colspan = 3,
    fig_name = Bx_ft.data_name,
    plot_path = f"{PATH_TO_PLOT}/bfield/{Bx_ft.data_name}",
    plot_name = "a) {}".format(r"$\delta B_x/B_0$"),
    limits = [(None,None), (None,None)],
    labels = [r"$x/\lambda_{si}$", r"$y/\lambda_{si}$"],
    cmap = "turbo",
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = (10.0,2.0),
    minor_loc = (1.0,1.0),
    levels = (None,None),
    # levels = (-0.1, 0.1),
    cbar_label = "",
    cbar_size = "2%",
    cbar_pad = "2%",
)

ft_plot_params = PlotParams(
    colspan = 1,
    fig_name = Bx_ft.data_name,
    plot_path = f"{PATH_TO_PLOT}/bfieldns/{Bx_ft.data_name}",
    plot_name = None,
    limits = [(None,None), (None,None)],
    # limits = [(None,10), (-10,10)],
    labels = [r"$k_x\lambda_{se}$", r"$k_y\lambda_{se}$"],
    cmap = cmap_ft,
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = (None, None),
    minor_loc = (None, None),
    # levels = (-40, -38),
    levels = (None, None),
    cbar_label = r"$dP(B_x)/dk$",
    cbar_size = "10%",
    cbar_pad = "6%",
)

Bx_ft.plot_params = plot_params
Bx_ft.plot_params_ft = ft_plot_params

# Create a dictionary containing all maps
fourier_bfield_dict = {
    "bfield_ft": [Bx_ft],
}