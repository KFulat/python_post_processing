from copy import deepcopy

from shock_visualization.constants import (
    PATH_TO_RESULT, PATH_TO_PLOT, N0, b0, e0, B0, E0, V0, C
)
from shock_visualization.quantities import CurrentTotal, Field, PlotParams

filter_sigma = 2.0

Jix = Field(PATH_TO_RESULT, "currxiresR", N0*V0, filter_sigma, N0*V0, False, True, True)
Jiy = Field(PATH_TO_RESULT, "curryiresR", N0*V0, filter_sigma, 0.0, False, True, False)
Jiz = Field(PATH_TO_RESULT, "currziresR", N0*V0, filter_sigma, 0.0, False, True, False)

Jex = Field(PATH_TO_RESULT, "currxeresR", N0*V0, filter_sigma, N0*V0, False, True, True)
Jey = Field(PATH_TO_RESULT, "curryeresR", N0*V0, filter_sigma, 0.0, False, True, False)
Jez = Field(PATH_TO_RESULT, "currzeresR", N0*V0, filter_sigma, 0.0, False, True, False)

Jx = CurrentTotal(1.0,0.0,0.0,Jix,Jex)
Jy = CurrentTotal(1.0,0.0,0.0,Jiy,Jey)
Jz = CurrentTotal(1.0,0.0,0.0,Jiz,Jez)

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

# Initialize plot parameters for density and field maps
curr_plot_params = PlotParams(
    colspan = colspan,
    fig_name = "step",
    plot_path = f"{PATH_TO_PLOT}/curr",
    plot_name = "c) {}".format(r"$J_z/J_0$"),
    limits = lims,
    labels = labels,
    cmap = cmap,
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = major_loc,
    minor_loc = minor_loc,
    levels = (-0.02, 0.02),
    # levels = (None, None),
    cbar_label = "",
    cbar_size=cbar_size,
    cbar_pad=cbar_pad,
)

# Attach plot parameters to quantities
Jz.plot_params = deepcopy(curr_plot_params)
Jx.plot_params = deepcopy(curr_plot_params)
# Jx.plot_params.levels = (-0.2,0.2)
Jx.plot_params.plot_name = "a) {}".format(r"$J_x/J_0$")
Jx.plot_params.hide_xlabels = True
Jx.plot_params.labels[0] = ""
Jy.plot_params = deepcopy(curr_plot_params)
Jy.plot_params.plot_name = "b) {}".format(r"$J_y/J_0$")
Jy.plot_params.hide_xlabels = True
Jy.plot_params.labels[0] = ""

# Create a dictionary containing all maps
curr_linear_dict = {
    "curr": [Jx, Jy, Jz],
}
