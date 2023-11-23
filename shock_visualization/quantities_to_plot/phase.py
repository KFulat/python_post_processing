from copy import deepcopy

from shock_visualization.constants import PATH_TO_RESULT, PATH_TO_PLOT
from shock_visualization.quantities import Phase, PlotParams

FILTER_SIGMA = 0.5

Pxi = Phase(PATH_TO_RESULT, "pxiR", 1.0, FILTER_SIGMA, True, False)
Pyi = Phase(PATH_TO_RESULT, "pyiR", 1.0, FILTER_SIGMA, True, False)
Pzi = Phase(PATH_TO_RESULT, "pziR", 1.0, FILTER_SIGMA, True, False)

Pxe = Phase(PATH_TO_RESULT, "pxeR", 1.0, FILTER_SIGMA, True, False)
Pye = Phase(PATH_TO_RESULT, "pyeR", 1.0, FILTER_SIGMA, True, False)
Pze = Phase(PATH_TO_RESULT, "pzeR", 1.0, FILTER_SIGMA, True, False)

# Phase plot parameters
xlims = (0, 100)
xlabel = r"$x/\lambda_{si}$"
levels = (-1.5, 4.0)
colspan = 5
fig_name = "step"

# Initialize plot parameters for density and field maps
ion_plot_params = PlotParams(
    colspan = colspan,
    fig_name = fig_name,
    plot_path = f"{PATH_TO_PLOT}/phase_ion",
    plot_name = "c) log{}".format(r"$N_i$"),
    limits = [xlims, (-0.65, 0.65)],
    labels = [xlabel,"{}".format(r"$p_{z,i}/m_i c$")],
    cmap = "magma",
    hide_xlabels = True,
    hide_ylabels = False,
    major_loc = (20.0,0.2),
    minor_loc = (1.0,0.05),
    levels = levels,
    cbar_label = "",
)

ele_plot_params = PlotParams(
    colspan = colspan,
    fig_name = fig_name,
    plot_path = f"{PATH_TO_PLOT}/phase_ele",
    plot_name = "c) log{}".format(r"$N_e$"),
    limits = [xlims, (-3.9, 3.9)],
    labels = [xlabel,"{}".format(r"$p_{z,e}/m_e c$")],
    cmap = "magma",
    hide_xlabels = True,
    hide_ylabels = False,
    major_loc = (20.0,2.0),
    minor_loc = (1.0,0.2),
    levels = levels,
    cbar_label = "",
)

# Attach plot parameters to quantities
Pxi.plot_params = deepcopy(ion_plot_params)
Pyi.plot_params = deepcopy(ion_plot_params)
Pzi.plot_params = deepcopy(ion_plot_params)
Pxi.plot_params.plot_name = "a) log{}".format(r"$N_i$")
Pyi.plot_params.plot_name = "b) log{}".format(r"$N_i$")
Pxi.plot_params.labels = ["", "{}".format(r"$p_{x,i}/m_i c$")]
Pyi.plot_params.labels = ["", "{}".format(r"$p_{y,i}/m_i c$")]
Pzi.plot_params.hide_xlabels = False

Pxe.plot_params = deepcopy(ele_plot_params)
Pye.plot_params = deepcopy(ele_plot_params)
Pze.plot_params = deepcopy(ele_plot_params)
Pxe.plot_params.plot_name = "a) log{}".format(r"$N_e$")
Pye.plot_params.plot_name = "b) log{}".format(r"$N_e$")
Pxe.plot_params.labels = ["", "{}".format(r"$p_{x,e}/m_e c$")]
Pye.plot_params.labels = ["", "{}".format(r"$p_{y,e}/m_e c$")]
Pze.plot_params.hide_xlabels = False

# Create a dictionary containing phase plots
phase_dict = {
    "phase_ion": [Pxi, Pyi, Pzi],
    "phase_ele": [Pxe, Pye, Pze],
}
