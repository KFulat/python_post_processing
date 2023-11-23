"""Module for plotting basic shock quantities."""

import numpy as np
import matplotlib


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

from visualization_package.plot import FigPlot

# quantities_dict = {**maps_dict, **phase_dict, **maps_linear_dict}
quantities_dict = {**phase_dict, **maps_linear_dict}

matplotlib.pyplot.style.use(
    "./src/post_processing/python_post_processing/"
    + "shock_visualization/basic.mplstyle"
)

def prepare_component(comp, step):
    comp.data_load(step)
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

    for nstep in range(args.start, args.stop + args.step, args.step):
        nstep_string = format_step_string(nstep)
        # x_sh = shock_position_from_file(nstep)
        x_sh = shock_position_linear(nstep)
        for quantity in args.quantities:
            quantity_components = quantities_dict[quantity]
            plots = []

            if quantity in ("phase_ion", "phase_ele"):
                matplotlib.rcParams["image.aspect"] = 'auto'
            else:
                matplotlib.rcParams["image.aspect"] = 'equal'

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
