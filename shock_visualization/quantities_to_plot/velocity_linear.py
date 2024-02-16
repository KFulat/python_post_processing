from copy import deepcopy

from shock_visualization.constants import (
    PATH_TO_RESULT, PATH_TO_PLOT, N0, b0, e0, B0, E0, V0, C
)
from shock_visualization.quantities import Density, Field, Velocity, PlotParams

filter_sigma = 2.5

Ni = Density(PATH_TO_RESULT, "densiresR", 1.0, 0.0, 0.0)
Ne = Density(PATH_TO_RESULT, "denseresR", 1.0, 0.0, 0.0)

Jix = Field(PATH_TO_RESULT, "currxiresR", 1.0, 0.0, 0.0)
Jiy = Field(PATH_TO_RESULT, "curryiresR", 1.0, 0.0, 0.0)
Jiz = Field(PATH_TO_RESULT, "currziresR", 1.0, 0.0, 0.0)

Jex = Field(PATH_TO_RESULT, "currxeresR", 1.0, 0.0, 0.0)
Jey = Field(PATH_TO_RESULT, "curryeresR", 1.0, 0.0, 0.0)
Jez = Field(PATH_TO_RESULT, "currzeresR", 1.0, 0.0, 0.0)

Vix = Velocity(1.0, filter_sigma, 0.0, Jix, Ni)
Viy = Velocity(1.0, filter_sigma, 0.0, Jiy, Ni)
Viz = Velocity(1.0, filter_sigma, 0.0, Jiz, Ni)

Vex = Velocity(1.0, filter_sigma, 0.0, Jex, Ne)
Vey = Velocity(1.0, filter_sigma, 0.0, Jey, Ne)
Vez = Velocity(1.0, filter_sigma, 0.0, Jez, Ne)

# Common plot parameters for all maps
lims = [(0, 5.75), (0, 5.75)]
labels = [r"$x/\lambda_{si}$", r"$y/\lambda_{si}$"]
major_loc = (1.0,1.0)
minor_loc = (0.2,0.2)
colspan = 1
fig_name = "step"
cmap = "turbo"
cbar_size = "5%"
cbar_pad = "5%"

# Initialize plot parameters for velocity
vel_plot_params = PlotParams(
    colspan = colspan,
    fig_name = "step_ion",
    plot_path = f"{PATH_TO_PLOT}/vel",
    plot_name = "c) {}".format(r"$(v_{i,z}-v_{z0})/v_0$"),
    limits = lims,
    labels = labels,
    cmap = cmap,
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = major_loc,
    minor_loc = minor_loc,
    levels = (-0.0015, 0.0015),
    # levels = (None, None),
    cbar_label = "",
    cbar_size=cbar_size,
    cbar_pad=cbar_pad,
)

Viz.plot_params = deepcopy(vel_plot_params)
Vix.plot_params = deepcopy(vel_plot_params)
Vix.plot_params.levels = (V0-0.003,V0+0.003)
Vix.plot_params.plot_name = "a) {}".format(r"$(v_{i,x}-v_{x0})/v_0$")
Vix.plot_params.hide_xlabels = True
Vix.plot_params.labels[0] = ""
Viy.plot_params = deepcopy(vel_plot_params)
Viy.plot_params.plot_name = "b) {}".format(r"$(v_{i,y}-v_{y0})/v_0$")
Viy.plot_params.hide_xlabels = True
Viy.plot_params.labels[0] = ""

vel_plot_params.fig_name = "step_ele"
vel_plot_params.plot_name = "a) {}".format(r"$(v_{e,z}-v_{z0})/v_0$")
Vez.plot_params = deepcopy(vel_plot_params)
Vex.plot_params = deepcopy(vel_plot_params)
Vex.plot_params.levels = (V0-0.003,V0+0.003)
Vex.plot_params.plot_name = "a) {}".format(r"$(v_{e,x}-v_{x0})/v_0$")
Vex.plot_params.hide_xlabels = True
Vex.plot_params.labels[0] = ""
Vey.plot_params = deepcopy(vel_plot_params)
Vey.plot_params.plot_name = "b) {}".format(r"$(v_{e,y}-v_{y0})/v_0$")
Vey.plot_params.hide_xlabels = True
Vey.plot_params.labels[0] = ""

# Create a dictionary containing all maps
vel_linear_dict = {
    "vel_ion": [Vix, Viy, Viz],
    "vel_ele": [Vex, Vey, Vez]
}
