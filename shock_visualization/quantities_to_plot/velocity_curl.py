from copy import deepcopy

from shock_visualization.constants import (
    PATH_TO_RESULT, PATH_TO_PLOT, N0, b0, e0, B0, E0, V0, C
)
from shock_visualization.quantities import (
    Density, Field, Velocity, CurlOfField, PlotParams
)

filter_sigma = 2.0

Ni = Density(PATH_TO_RESULT, "densiresR", 1.0, 0.0, 0.0)
Ne = Density(PATH_TO_RESULT, "denseresR", 1.0, 0.0, 0.0)

Jix = Field(PATH_TO_RESULT, "currxiresR", 1.0, 0.0, 0.0)
Jiy = Field(PATH_TO_RESULT, "curryiresR", 1.0, 0.0, 0.0)
Jiz = Field(PATH_TO_RESULT, "currziresR", 1.0, 0.0, 0.0)

Jex = Field(PATH_TO_RESULT, "currxeresR", 1.0, 0.0, 0.0)
Jey = Field(PATH_TO_RESULT, "curryeresR", 1.0, 0.0, 0.0)
Jez = Field(PATH_TO_RESULT, "currzeresR", 1.0, 0.0, 0.0)

Vix = Velocity(V0, filter_sigma, V0, Jix, Ni, norm=True, extract=True)
Viy = Velocity(V0, filter_sigma, 0.0, Jiy, Ni, norm=True, extract=True)
Viz = Velocity(V0, filter_sigma, 0.0, Jiz, Ni, norm=True, extract=True)

Vex = Velocity(V0, filter_sigma, V0, Jex, Ne, norm=True, extract=True)
Vey = Velocity(V0, filter_sigma, 0.0, Jey, Ne, norm=True, extract=True)
Vez = Velocity(V0, filter_sigma, 0.0, Jez, Ne, norm=True, extract=True)

curlVi = CurlOfField(abs(V0), 1.0, 0.0, Vix, Viy, Viz, norm=True)
curlVe = CurlOfField(abs(V0), 1.0, 0.0, Vex, Vey, Vez, norm=True)

# Common plot parameters for all maps
lims = [(0, 5.75), (0, 5.75)]
labels = [r"$x/\lambda_{si}$", r"$y/\lambda_{si}$"]
major_loc = (1.0,1.0)
minor_loc = (0.2,0.2)
colspan = 1
cmap = "turbo"
cbar_size = "5%"
cbar_pad = "5%"

# Initialize plot parameters for velocity
curl_plot_params = PlotParams(
    colspan = colspan,
    fig_name = "step_curl_ion",
    plot_path = f"{PATH_TO_PLOT}/vel/curl",
    plot_name = "{}".format(r"$|\nabla \times \delta v_i|/v_0$"),
    limits = lims,
    labels = labels,
    cmap = cmap,
    hide_xlabels = False,
    hide_ylabels = False,
    major_loc = major_loc,
    minor_loc = minor_loc,
    levels = (0.5e-3, 4.5e-3),
    # levels = (None, None),
    cbar_label = "",
    cbar_size=cbar_size,
    cbar_pad=cbar_pad,
)

curlVi.plot_params = deepcopy(curl_plot_params)

curl_plot_params.fig_name = "step_ele"
curl_plot_params.plot_name = "{}".format(r"$|\nabla \times \delta v_e|/v_0$")
curlVe.plot_params = deepcopy(curl_plot_params)

# Create a dictionary containing all maps
curl_vel_dict = {
    "curl_vel_ion": [curlVi],
    "curl_vel_ele": [curlVe]
}
