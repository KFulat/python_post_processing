from copy import deepcopy

from shock_visualization.constants import (
    PATH_TO_RESULT, PATH_TO_PLOT, N0, b0, e0, B0, E0, V0, C
)
from shock_visualization.quantities import Density, Field, PlotParams

filter_sigma = 1.0

Ni = Density(PATH_TO_RESULT, "densiresR", N0, filter_sigma, 0.0, False, True, False)
Ne = Density(PATH_TO_RESULT, "denseresR", N0, filter_sigma, 0.0, False, True, False)
Bx = Field(PATH_TO_RESULT, "bxres", b0, filter_sigma, B0[0], False, True, True)
By = Field(PATH_TO_RESULT, "byres", b0, filter_sigma, B0[1], False, True, True)
Bz = Field(PATH_TO_RESULT, "bzres", b0, filter_sigma, B0[2], False, True, True)
Ex = Field(PATH_TO_RESULT, "exres", e0, filter_sigma, E0[0], False, True, True)
Ey = Field(PATH_TO_RESULT, "eyres", e0, filter_sigma, E0[1], False, True, True)
Ez = Field(PATH_TO_RESULT, "ezres", e0, filter_sigma, E0[2], False, True, True)

Jix = Field(PATH_TO_RESULT, "currxiresR", N0*V0, filter_sigma, N0*V0, False, True, True)
Jiy = Field(PATH_TO_RESULT, "curryiresR", N0*V0, filter_sigma, 0.0, False, True, False)
Jiz = Field(PATH_TO_RESULT, "currziresR", N0*V0, filter_sigma, 0.0, False, True, False)

Jex = Field(PATH_TO_RESULT, "currxeresR", N0*V0, filter_sigma, N0*V0, False, True, True)
Jey = Field(PATH_TO_RESULT, "curryeresR", N0*V0, filter_sigma, 0.0, False, True, False)
Jez = Field(PATH_TO_RESULT, "currzeresR", N0*V0, filter_sigma, 0.0, False, True, False)

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
dens_plot_params = PlotParams(
    colspan = colspan,
    fig_name = fig_name,
    plot_path = f"{PATH_TO_PLOT}/dens",
    plot_name = "b) {}".format(r"$N_e/N_0$"),
    limits = lims,
    labels = labels,
    cmap = cmap,
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = major_loc,
    minor_loc = minor_loc,
    levels = (1.0-0.15, 1.0+0.15),
    cbar_label = "",
    cbar_size=cbar_size,
    cbar_pad=cbar_pad,
)

bfield_plot_params = PlotParams(
    colspan = colspan,
    fig_name = fig_name,
    plot_path = f"{PATH_TO_PLOT}/bfield",
    plot_name = "c) {}".format(r"$(B_z-B_{z0})/B_0$"),
    limits = lims,
    labels = labels,
    cmap = cmap,
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = major_loc,
    minor_loc = minor_loc,
    levels = (-1e-2, 1e-2),
    # levels = (None, None),
    cbar_label = "",
    cbar_size=cbar_size,
    cbar_pad=cbar_pad,
)

efield_plot_params = PlotParams(
    colspan = colspan,
    fig_name = fig_name,
    plot_path = f"{PATH_TO_PLOT}/efield",
    plot_name = "c) {}".format(r"$(E_z-E_{z0})/E_0$"),
    limits = lims,
    labels = labels,
    cmap = cmap,
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = major_loc,
    minor_loc = minor_loc,
    levels = (-0.2, 0.2),
    # levels = (None, None),
    cbar_label = "",
    cbar_size=cbar_size,
    cbar_pad=cbar_pad,
)

curr_plot_params = PlotParams(
    colspan = colspan,
    fig_name = "step_ion",
    plot_path = f"{PATH_TO_PLOT}/curr",
    plot_name = "c) {}".format(r"$(J_{i,z}-J_{z0})/J_0$"),
    limits = lims,
    labels = labels,
    cmap = cmap,
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = major_loc,
    minor_loc = minor_loc,
    levels = (-0.01, 0.01),
    cbar_label = "",
    cbar_size=cbar_size,
    cbar_pad=cbar_pad,
)

# Attach plot parameters to quantities
Ne.plot_params = dens_plot_params
Ni.plot_params = deepcopy(dens_plot_params)
Ni.plot_params.plot_name = "a) {}".format(r"$N_i/N_0$")
Ni.plot_params.labels[0] = ""
Ni.plot_params.hide_xlabels = True

Bz.plot_params = bfield_plot_params
Bx.plot_params = deepcopy(bfield_plot_params)
Bx.plot_params.plot_name = "a) {}".format(r"$(B_x-B_{x0})/B_0$")
Bx.plot_params.hide_xlabels = True
Bx.plot_params.labels[0] = ""
By.plot_params = deepcopy(bfield_plot_params)
By.plot_params.plot_name = "b) {}".format(r"$(B_y-B_{y0})/B_0$")
By.plot_params.hide_xlabels = True
By.plot_params.labels[0] = ""

Ez.plot_params = efield_plot_params
Ex.plot_params = deepcopy(efield_plot_params)
Ex.plot_params.plot_name = "a) {}".format(r"$(E_x-E_{x0})/E_0$")
Ex.plot_params.hide_xlabels = True
Ex.plot_params.labels[0] = ""
Ey.plot_params = deepcopy(efield_plot_params)
Ey.plot_params.plot_name = "b) {}".format(r"$(E_y-E_{y0})/E_0$")
Ey.plot_params.hide_xlabels = True
Ey.plot_params.labels[0] = ""

Jiz.plot_params = deepcopy(curr_plot_params)
Jix.plot_params = deepcopy(curr_plot_params)
Jix.plot_params.levels = (-0.2,0.2)
Jix.plot_params.plot_name = "a) {}".format(r"$(J_{i,x}-J_{x0})/J_0$")
Jix.plot_params.hide_xlabels = True
Jix.plot_params.labels[0] = ""
Jiy.plot_params = deepcopy(curr_plot_params)
Jiy.plot_params.plot_name = "b) {}".format(r"$(J_{i,y}-J_{y0})/J_0$")
Jiy.plot_params.hide_xlabels = True
Jiy.plot_params.labels[0] = ""

curr_plot_params.fig_name = "step_ele"
curr_plot_params.plot_name = "a) {}".format(r"$(J_{e,z}-J_{z0})/J_0$")
Jez.plot_params = deepcopy(curr_plot_params)
Jex.plot_params = deepcopy(curr_plot_params)
Jex.plot_params.levels = (-0.2,0.2)
Jex.plot_params.plot_name = "a) {}".format(r"$(J_{e,x}-J_{x0})/J_0$")
Jex.plot_params.hide_xlabels = True
Jex.plot_params.labels[0] = ""
Jey.plot_params = deepcopy(curr_plot_params)
Jey.plot_params.plot_name = "b) {}".format(r"$(J_{e,y}-J_{y0})/J_0$")
Jey.plot_params.hide_xlabels = True
Jey.plot_params.labels[0] = ""

# Create a dictionary containing all maps
maps_linear_dict = {
    "dens": [Ni, Ne],
    "bfield": [Bx, By, Bz],
    "efield": [Ex, Ey, Ez],
    "curr_ion": [Jix, Jiy, Jiz],
    "curr_ele": [Jex, Jey, Jez]
}
