"""Module for plotting basic shock quantities."""

import numpy as np
import matplotlib
from copy import deepcopy


from shock_visualization.tools import (
    make_fig_header_shock_pos,
    make_fig_header,
    format_step_string,
    # load_comand_line,
    # shock_position_from_file,
    shock_position_linear,
    parse_arguments,
)
from shock_visualization.quantities_to_plot.maps import maps_dict
from shock_visualization.quantities_to_plot.phase import phase_dict
from shock_visualization.quantities_to_plot.maps_linear import (
    maps_linear_dict
)

from shock_visualization.constants import (
    PATH_TO_RESULT, PATH_TO_PLOT, N0, b0, e0, B0, E0, V0, C
)
from shock_visualization.quantities import Density, Field, PlotParams

from visualization_package.plot import FigPlot

# quantities_dict = {**maps_dict, **phase_dict, **maps_linear_dict}
# quantities_dict = {**phase_dict, **maps_linear_dict}

matplotlib.pyplot.style.use(
    "shock_visualization/basic.mplstyle"
)

def prepare_component(comp, step):
    # comp.data_load(step)
    comp.data_extract()
    comp.data_normalize()
    comp.data_logscale()
    comp.set_ticks()

def create_fig(comp, nstep, nstep_string, x_sh, quantity, args, plots):
    if args.subdirectory:
        fig_path = (
            f"{comp.plot_params.plot_path}/"
            + f"{args.subdirectory}/"
        )
    else:
        fig_path = f"{comp.plot_params.plot_path}/"
    fig_path = (
        fig_path
        + f"{comp.plot_params.fig_name}_{nstep_string}"
        + f".{args.filetype}"
    )

    fig = FigPlot(
        fig_path,
        plots,
        (len(plots), comp.plot_params.colspan),
        size=(1.5*3.7, 1.5*3*2.825),
        hspace=0.15,
        dpi=args.dpi,
    )

    fig_header = make_fig_header(nstep)

    if quantity == "phase_ele":
        fig.plots[0].axis.text(
            fig.plots[0].lims[0][0],
            1.2*fig.plots[0].lims[1][1],
            fig_header,
            fontsize=22
        )
    else:
        fig.plots[0].axis.text(
            fig.plots[0].lims[0][0],
            1.07*fig.plots[0].lims[1][1],
            fig_header,
            fontsize=22
        )
    return fig

if __name__ == "__main__":
    args = parse_arguments()

    Vix = Field("","",1.0,0.0,0.0)
    Viy = Field("","",1.0,0.0,0.0)
    Viz = Field("","",1.0,0.0,0.0)

    Vex = Field("","",1.0,0.0,0.0)
    Vey = Field("","",1.0,0.0,0.0)
    Vez = Field("","",1.0,0.0,0.0)

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

    curr_plot_params = PlotParams(
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

    Viz.plot_params = deepcopy(curr_plot_params)
    Vix.plot_params = deepcopy(curr_plot_params)
    Vix.plot_params.levels = (V0-0.003,V0+0.003)
    Vix.plot_params.plot_name = "a) {}".format(r"$(v_{i,x}-v_{x0})/v_0$")
    Vix.plot_params.hide_xlabels = True
    Vix.plot_params.labels[0] = ""
    Viy.plot_params = deepcopy(curr_plot_params)
    Viy.plot_params.plot_name = "b) {}".format(r"$(v_{i,y}-v_{y0})/v_0$")
    Viy.plot_params.hide_xlabels = True
    Viy.plot_params.labels[0] = ""

    curr_plot_params.fig_name = "step_ele"
    curr_plot_params.plot_name = "a) {}".format(r"$(v_{e,z}-v_{z0})/v_0$")
    Vez.plot_params = deepcopy(curr_plot_params)
    Vex.plot_params = deepcopy(curr_plot_params)
    Vex.plot_params.levels = (V0-0.003,V0+0.003)
    Vex.plot_params.plot_name = "a) {}".format(r"$(v_{e,x}-v_{x0})/v_0$")
    Vex.plot_params.hide_xlabels = True
    Vex.plot_params.labels[0] = ""
    Vey.plot_params = deepcopy(curr_plot_params)
    Vey.plot_params.plot_name = "b) {}".format(r"$(v_{e,y}-v_{y0})/v_0$")
    Vey.plot_params.hide_xlabels = True
    Vey.plot_params.labels[0] = ""

    filter_sigma = 2.5

    Ni = Density(
        PATH_TO_RESULT, "densiresR", N0, filter_sigma, 0.0,
        False, False, False
    )
    Ne = Density(
        PATH_TO_RESULT, "denseresR", N0, filter_sigma, 0.0,
        False, False, False
    )
    Jix = Field(
        PATH_TO_RESULT, "currxiresR", N0*V0, filter_sigma, N0*V0,
        False, False, False
    )
    Jiy = Field(
        PATH_TO_RESULT, "curryiresR", N0*V0, filter_sigma, 0.0,
        False, False, False
    )
    Jiz = Field(
        PATH_TO_RESULT, "currziresR", N0*V0, filter_sigma, 0.0,
        False, False, False
    )
    Jex = Field(
        PATH_TO_RESULT, "currxeresR", N0*V0, filter_sigma, N0*V0,
        False, False, False
    )
    Jey = Field(
        PATH_TO_RESULT, "curryeresR", N0*V0, filter_sigma, 0.0,
        False, False, False
    )
    Jez = Field(
        PATH_TO_RESULT, "currzeresR", N0*V0, filter_sigma, 0.0,
        False, False, False
    )

    for nstep in range(args.start, args.stop + args.step, args.step):
        nstep_string = format_step_string(nstep)
        # x_sh = shock_position_from_file(nstep)
        x_sh = shock_position_linear(nstep)

        for quantity in [Ni,Ne,Jix,Jiy,Jiz,Jex,Jey,Jez]:
            quantity.data_load(nstep)

        Vix.data = Jix.data / Ni.data
        Viy.data = Jiy.data / Ni.data
        Viz.data = Jiz.data / Ni.data

        Vex.data = Jex.data / Ne.data
        Vey.data = Jey.data / Ne.data
        Vez.data = Jez.data / Ne.data

        vel_dict = {
            "vel_ion": [Vix, Viy, Viz],
            "vel_ele": [Vex, Vey, Vez]
        }

        quantities_dict = {**vel_dict}

        for quantity in args.quantities:
            quantity_components = quantities_dict[quantity]
            plots = []

            for i, component in enumerate(quantity_components):
                prepare_component(component, nstep)

                if args.levels:
                    component.plot_params.levels = args.levels
                if args.xlimits:
                    component.plot_params.limits[0] = args.xlimits
                if args.ylimits:
                    component.plot_params.limits[1] = args.ylimits
                if args.follow_shock:
                    a = args.follow_shock[0]
                    b = args.follow_shock[1]
                    component.plot_params.limits[0] = (x_sh+a, x_sh+b)

                component.add_plot((i,0))
                plots.append(component.plot)

            fig = create_fig(component, nstep, nstep_string, x_sh,
                             quantity, args, plots)
            fig.save()
