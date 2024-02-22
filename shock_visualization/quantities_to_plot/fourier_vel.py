from copy import deepcopy
from shock_visualization.constants import (
    PATH_TO_RESULT, PATH_TO_PLOT, N0, b0, e0, B0, E0, V0
)
from shock_visualization.quantities import (
    Density, Fourier, PlotParams, Field, Velocity, Magnitude, PlotParams1D
)
from shock_visualization.mycolormaps import cmap_ft

filter_sigma = 0.0

Ni = Density(PATH_TO_RESULT, "densiresR", 1.0, 0.0, 0.0)
Ne = Density(PATH_TO_RESULT, "denseresR", 1.0, 0.0, 0.0)

Jix = Field(PATH_TO_RESULT, "currxiresR", 1.0, 0.0, 0.0)
Jiy = Field(PATH_TO_RESULT, "curryiresR", 1.0, 0.0, 0.0)
Jiz = Field(PATH_TO_RESULT, "currziresR", 1.0, 0.0, 0.0)

Jex = Field(PATH_TO_RESULT, "currxeresR", 1.0, 0.0, 0.0)
Jey = Field(PATH_TO_RESULT, "curryeresR", 1.0, 0.0, 0.0)
Jez = Field(PATH_TO_RESULT, "currzeresR", 1.0, 0.0, 0.0)

Vix = Velocity(V0, 0.0, V0,  Jix, Ni, norm=True, extract=True)
Viy = Velocity(V0, 0.0, 0.0, Jiy, Ni, norm=True, extract=True)
Viz = Velocity(V0, 0.0, 0.0, Jiz, Ni, norm=True, extract=True)

Vex = Velocity(V0, 0.0, V0,  Jex, Ne, extract=True)
Vey = Velocity(V0, 0.0, 0.0, Jey, Ne, extract=True)
Vez = Velocity(V0, 0.0, 0.0, Jez, Ne, extract=True)

Vi = Magnitude(V0, filter_sigma, 0.0, Vix, Viy, Viz)
Ve = Magnitude(V0, filter_sigma, 0.0, Vex, Vey, Vez)

Vi_ft = Fourier(Vi, "step_ion_ft", V0, 0, 0.0, 4)
Ve_ft = Fourier(Vi, "step_ele_ft", V0, 0, 0.0, 4)

plot_params = PlotParams(
    colspan = 3,
    fig_name = Vi_ft.data_name,
    plot_path = f"{PATH_TO_PLOT}/vel/fourier",
    plot_name = "{}".format(r"$\delta v_i/v_0$"),
    limits = [(None,None), (None,None)],
    labels = [r"$x/\lambda_{si}$", r"$y/\lambda_{si}$"],
    cmap = "turbo",
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = (10.0,2.0),
    minor_loc = (1.0,1.0),
    levels = (None,None),
    # levels = (-0.15, 0.15),
    cbar_label = "",
    cbar_size = "2%",
    cbar_pad = "2%",
)

ft_plot_params = PlotParams(
    colspan = 1,
    fig_name = Vi_ft.data_name,
    plot_path = f"{PATH_TO_PLOT}/vel/fourier",
    plot_name = None,
    # limits = [(None,None), (None,None)],
    limits = [(None,1.5), (-1.5,1.5)],
    labels = [r"$k_x\lambda_{se}$", r"$k_y\lambda_{se}$"],
    cmap = "turbo",
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = (None, None),
    minor_loc = (None, None),
    # levels = (-9, -6),
    levels = (None, None),
    cbar_label = r"$dP(\delta v_i/v_0)/dk$",
    cbar_size = "10%",
    cbar_pad = "6%",
)

ft1d_plot_params = PlotParams1D(
    colspan = 1,
    fig_name = "{}1D".format(Vi_ft.data_name,),
    plot_path = f"{PATH_TO_PLOT}/vel/fourier",
    plot_name = None,
    limits = [(1e-1,1.1e1),(1e-9,1e-4)],
    # limits = [(None,None), (None,None)],
    labels = [r"$k\lambda_{se}$", r"$P_{\delta v_i}(k)$"],
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = (None, None),
    minor_loc = (None, None),
    legend = False,
    legend_position = "",
)

Vi_ft.plot_params = plot_params
Vi_ft.plot_params_ft = ft_plot_params
Vi_ft.plot_params_ft1d = deepcopy(ft1d_plot_params)

# Create a dictionary containing all maps
fourier_vel_dict = {
    "vel_ion_ft": [Vi_ft],
}