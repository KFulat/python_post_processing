from copy import deepcopy

from shock_visualization.constants import (
    PATH_TO_RESULT, PATH_TO_PLOT, N0, b0, e0, B0, E0, V0, C
)
from shock_visualization.quantities import Magnitude, Density, Field, PlotParams, ProfileX, PlotParams1D

filter_sigma = 2.0

Ni = Density(PATH_TO_RESULT, "densiresR", 1.0, 0.0, 0.0)
Ne = Density(PATH_TO_RESULT, "denseresR", 1.0, 0.0, 0.0)

Bx = Field(PATH_TO_RESULT, "bxres", 1.0, 0.0, 0.0)
By = Field(PATH_TO_RESULT, "byres", 1.0, 0.0, 0.0)
Bz = Field(PATH_TO_RESULT, "bzres", 1.0, 0.0, 0.0)

Ex = Field(PATH_TO_RESULT, "exres", 1.0, 0.0, 0.0)
Ey = Field(PATH_TO_RESULT, "eyres", 1.0, 0.0, 0.0)
Ez = Field(PATH_TO_RESULT, "ezres", 1.0, 0.0, 0.0)

B = Magnitude(1.0, 0.0, 0.0, Bx, By, Bz)
E = Magnitude(1.0, 0.0, 0.0, Ex, Ey, Ez)

Nprof = ProfileX("", "step_prof", N0, filter_sigma, [Ni, Ne],     norm=True, abs=True)
Bprof = ProfileX("", "step_prof", b0, filter_sigma, [B, Bx, By, Bz], norm=True, abs=True)
Eprof = ProfileX("", "step_prof", e0, filter_sigma, [E, Ex, Ey, Ez], norm=True, abs=True)

dens_plot_params = PlotParams1D(
    colspan = 3,
    fig_name = "step_prof",
    plot_path = "../profiles",
    plot_name = [r"$N_i/N_0$", r"$N_i/N_0$"],
    limits = [(0, 80), (0, 250/N0)],
    labels = ["", ""],
    hide_xlabels = True,
    hide_ylabels = False,
    major_loc = (None, None),
    minor_loc = (None, None),
    legend = True,
    legend_position = "upper right",
)
bfield_plot_params = PlotParams1D(
    colspan = 3,
    fig_name = "step_prof",
    plot_path = "../profiles",
    plot_name = [r"$B/B_0$", r"$|B_x|/B_0$", r"$|B_y|/B_0$", r"$|B_z|/B_0$"],
    limits = [(0, 80), (0, 20)],
    labels = ["", ""],
    hide_xlabels = True,
    hide_ylabels = False,
    major_loc = (None, None),
    minor_loc = (None, None),
    legend = True,
    legend_position = "upper right",
)
efield_plot_params = PlotParams1D(
    colspan = 3,
    fig_name = "step_prof",
    plot_path = "../profiles",
    plot_name = [r"$E/E_0$",r"$|E_x|/E_0$", r"$|E_y|/E_0$", r"$|E_z|/E_0$"],
    # limits = [(1e-8, 1e-1), (1e-6, 1e+1)],
    limits = [(0, 80), (0, 7)],
    labels = [r"$x/\lambda_{si}$", ""],
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = (None, None),
    minor_loc = (None, None),
    legend = True,
    legend_position = "upper right",
)

Nprof.plot_params = deepcopy(dens_plot_params)
Bprof.plot_params = deepcopy(bfield_plot_params)
Eprof.plot_params = deepcopy(efield_plot_params)

# Create a dictionary containing all maps
profiles_dict = {
    "prof_all": [Nprof, Bprof, Eprof],
}
