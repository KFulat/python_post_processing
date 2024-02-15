"""Module for plotting particle spectra."""

import matplotlib
from cycler import cycler

from shock_visualization.tools import (
    make_fig_header,
    parse_arguments,
    format_step_string,
)
from shock_visualization.distrfit import MaxwellFit
from shock_visualization.constants import MI, ME, C

from visualization_package.plot import FigPlot

from shock_visualization.quantities_to_plot.spectra import spectra_dict
quantities_dict = {**spectra_dict}

matplotlib.pyplot.style.use(
    "shock_visualization/basic.mplstyle"
)

def create_fig(component, nstep_string):
    fig_path = f"{component.plot_params.plot_path}/{component.plot_params.fig_name}_{nstep_string}.png"
    fig = FigPlot(
        fig_path,
        plots,
        (1, component.plot_params.colspan),
        hspace=0.05,
        dpi=300,
        size=(5,4)
    )
    fig_header = make_fig_header(nstep)
    # fig.plots[0].axis.text(
    #     0, 1.07*fig.plots[0].lims[1][1], fig_header, fontsize=22
    # )
    from matplotlib.ticker import LogFormatter, LogFormatterMathtext
    for i, plot in enumerate(fig.plots):
        fig.plots[i].axis.set_xscale("log", base=10)
        fig.plots[i].axis.set_yscale("log", base=10)
        fig.plots[i].axis.xaxis.get_major_locator().set_params(numticks=5)
        fig.plots[i].axis.xaxis.get_minor_locator().set_params(numticks=10)
        fig.plots[i].axis.yaxis.get_major_locator().set_params(numticks=5)
    return fig

if __name__ == "__main__":
    args = parse_arguments()

    xL = 1.0
    xR = xL + 7.0

    matplotlib.rcParams["axes.prop_cycle"] = cycler('color',
        ['black', '#EE6677', '#EE6677', '#4477AA'])+cycler('linestyle', ['-', '--', '--', '--'])

    for nstep in range(args.start, args.stop + args.step, args.step):
        nstep_string = format_step_string(nstep)
        plots = []

        for quantity in args.quantities:
            quantity = quantities_dict[quantity]
            print(quantity)
            quantity.data_load(nstep, xL, xR)

            # Obtain N(\gamma-1)*(\gamma-1) from dN/d(\gamma-1)*(\gamma-1)
            # quantity.data = quantity.data * quantity.delta

            quantity.plot_params.plot_name = []
            # quantity.plot_params.plot_name.append("{}{:.0f}{}{:.0f}{}".format(
            #     r"$x=($", xL, r"$-$", xR, r"$)\lambda_{si}$"))
            
            if quantity.data_name == "elec":
                quantity.xpoints /= ME*C**2
                quantity.plot_params.plot_name.append("electrons")
            elif quantity.data_name == "ions":
                quantity.plot_params.plot_name.append("ions")

            quantity.plot_params.plot_name.append("thermal fit")

            if quantity.data_name == "elec":
                maxwell = MaxwellFit(
                    quantity.xpoints, quantity.data, 3*1e-4, 7*1e-3,
                    quantity.bin_min, quantity.delta
                )
                maxwell.fit_with_curvefit()
                T = ME*C**2/maxwell.popt[1]
                text_xpos = 2*1e-8
            elif quantity.data_name == "ions":
                maxwell = MaxwellFit(
                    quantity.xpoints, quantity.data, 1e-3, 1e-2,
                    quantity.bin_min, quantity.delta
                )
                maxwell.fit_with_curvefit()
                T = MI*C**2/maxwell.popt[1]
                text_xpos = 1.3*1e-6

            print(f"Maxwell fit: 1/B'={1.0/maxwell.popt[1]}")
            print(f"Maxwell fit: kT={T}")

            quantity.data = list([
                quantity.data, maxwell.yfit
            ])

            quantity.add_plot((0,0))
            plots.append(quantity.plot)
            # nstep_string = f"{nstep_string}_{int(xL)}_{int(xR)}"

            # plots[0].legend = False

        fig = create_fig(quantity, nstep_string)
        fig.plots[0].axis.text(
            text_xpos, 4*1e-2, "{}{:.2e}".format(r"$\frac{k_BT_{FIT}}{m_ec^2}=$", T/(ME*C**2))
        )
        fig.save()
