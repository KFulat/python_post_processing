from shock_visualization.constants import DIGITS, LSI, STEP_INIT, ION_GYROTIME, C
from visualization_package.load import load_data_from_h5file

import sys
import numpy as np
from numba import njit
from argparse import ArgumentParser
from scipy.optimize import curve_fit

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("start", type=int)
    parser.add_argument("stop", type=int)
    parser.add_argument("step", type=int)

    parser.add_argument("quantities", nargs="*")

    parser.add_argument(
        "-log",
        "--logscale",
        action="store_true"
    )

    parser.add_argument(
        "--filetype",
        choices=["png", "pdf", "svg"],
        default="png"
    )

    parser.add_argument("--subdirectory")

    def str_to_tuple_int(str):
        list_of_str = str.split(",")
        list_of_int = [int(s) for s in list_of_str]
        return tuple(list_of_int)
    
    def str_to_tuple_float(str):
        list_of_str = str.split(",")
        list_of_int = [float(s) for s in list_of_str]
        return tuple(list_of_int)

    parser.add_argument(
        "--follow_shock",
        nargs=2,
        type=float,
        metavar=("A B"),
        help="take two floats to set xlimits to (x_sh+A, x_sh+B)"
    )

    parser.add_argument(
        "--xlimits",
        type=str_to_tuple_float,
        metavar=("XMIN,XMAX"),
        help="take xmin and xmax seperated by a comma"
    )

    parser.add_argument(
        "--ylimits",
        type=str_to_tuple_float,
        metavar=("YMIN,YMAX"),
        help="take ymin and ymax seperated by a comma"
    )

    parser.add_argument(
        "--levels",
        type=str_to_tuple_float,
        metavar=("MIN,MAX"),
        help="take min and max values for the colormap seperated by a comma"
    )

    parser.add_argument("--dpi", type=int)

    args = parser.parse_args()

    return args

def func_linear(x, a, b):
    return a*x+b

def linear_fit(x, y):
    """fit y = a*x + b"""
    popt, pcov = curve_fit(func_linear, x, y)
    perr = np.sqrt(np.diag(pcov))
    a = popt[0]
    b = popt[1]
    std_a = perr[0]
    x_mean = np.mean(x)
    sum = np.sum((x-x_mean)**2)
    # err_a = std_a / np.sqrt(sum)
    err_a = std_a
    return a, err_a, b

def load_comand_line():
    if len(sys.argv[1:]) >= 3:
        start, stop, step = [int(i) for i in sys.argv[1:4]]
        quantities_to_plot = sys.argv[4:]
    return start, stop, step, quantities_to_plot

def field_log(arr, floor):
    """
    Make logscale field array.
    """
    sign = np.sign(arr)
    arr_mask = np.ma.masked_equal(arr, 0.0)
    arr_mask = np.abs(arr_mask)
    arr_log = np.ma.log10(arr_mask)
    arr_log[arr_log < floor] = floor
    sign[arr_log < floor] = 0.0
    arr_log -= floor
    arr_log = arr_log.filled(0.0)
    arr_log *= sign
    return arr_log

def field_log_new(arr, floor):
    """
    Make logscale field array.
    """
    sign = np.sign(arr)
    arr_abs = np.abs(arr)
    arr_abs[arr_abs < floor] = floor
    arr_log = np.log10(arr_abs)
    arr_log -= np.log10(floor)
    arr_log *= sign
    return arr_log

def format_step_string(step):
    return "{:0{}d}".format(step, DIGITS)

def make_fig_header(step):
    step_str = "{}{}".format(r"$step$=", format_step_string(step-STEP_INIT))
    time_str = "{}{:3.2f} {}".format(r"$time=$", (step-STEP_INIT)/ION_GYROTIME, r"$[\Omega_i^{-1}]$")
    header = "{}, {}".format(step_str, time_str)
    return header

def make_fig_header_shock_pos(step, x_sh):
    header = make_fig_header(step)
    pos_str = "{}={:3.1f} {}".format(r"$x_{sh}$", x_sh, r"$[\lambda_{si}]$")
    header = "{}, {}".format(header, pos_str)
    return header

def load_movHR(step, data_name, path_to_result):
    step_digits = format_step_string(step)
    file_path = f"{path_to_result}/movHR_{step_digits}XY.h5"
    group_name = f"Step#{step}"
    group = load_data_from_h5file(file_path, group_name)
    data = group[group_name][data_name]
    return data

def load_mom_distr(step, data_name, path_to_result):
    step_digits = format_step_string(step)
    file_path = f"{path_to_result}/mom_distr_{step_digits}.h5"
    group = load_data_from_h5file(file_path, data_name)
    return group

def load_ekin(step, data_name, path_to_result):
    step_digits = format_step_string(step)
    file_path = f"{path_to_result}/ekin_{step_digits}.h5"
    group = load_data_from_h5file(file_path, data_name)
    return group

def load_phase(step, data_name, path_to_result):
    step_digits = format_step_string(step)
    file_path = f"{path_to_result}/phase_{step_digits}.h5"
    group_name = f"Step#{step}"
    group = load_data_from_h5file(file_path, group_name)
    data = group[group_name][data_name]
    return data   

def load_Npx(step, path_to_result):
    step_digits = format_step_string(step)
    file_path = f"{path_to_result}/movHR_{step_digits}XY.h5"
    group_name = f"Step#{step}"
    group = load_data_from_h5file(file_path, group_name)
    Npx = int(group[group_name]['attrs']['Npx'])
    return Npx

def shock_position_from_file(step):
    shock_pos = load_data_from_h5file("../shock_pos_file.h5", "shock_pos")
    shock_steps = load_data_from_h5file("../shock_pos_file.h5", "steps")
    index = np.where(shock_steps == step)[0][0]
    x_shock = shock_pos[index]
    return x_shock

def shock_position_linear(step):
    time = step-STEP_INIT
    v_sh = 0.06736*C
    x0 = -1165.0
    x_shock = v_sh*time + x0 
    return x_shock/LSI

@njit
def split_arr_into_list(arr, n, m):
    a = arr.shape[0]//n
    b = arr.shape[1]//m
    split = np.zeros((n, m*a*b))
    print(split.shape)
    index = 0
    for i in range(a):
        for j in range(b):
            split_arr = arr[i*n:(i+1)*n, j*m:(j+1)*m]
            split[:, index*m:(index+1)*m] = split_arr
            index += 1
    return split

@njit
def mean_stdev_list(list, m):
    n = list.shape[1] // m
    mean = np.zeros(n)
    sem = np.zeros(n)
    sd = np.zeros(n)
    for i in range(n):
        arr = list[:, i*m:(i+1)*m]
        mean_arr = np.mean(arr)
        std_arr = np.std(arr)
        mean_err_arr = std_arr / np.sqrt(arr.shape[0]*arr.shape[1])

        mean[i] = mean_arr
        sd[i] = std_arr
        sem[i] = mean_err_arr
    return mean, sem, sd

@njit
def electrostatic_field_from_charge(q, ncell=100):
    DX = 1.0
    DY = 1.0
    q_shape = q.shape
    m = q_shape[0]
    n = q_shape[1]
    Ex = np.zeros(q_shape)
    Ey = np.zeros(q_shape)
    
    for i in range(m):
        iimin = max(0, i-ncell)
        iimax = min(m-1, i+ncell)
        # print("i=", i, "iimin=", iimin, "iimax=", iimax)
        for j in range(n):
            for ii in range(iimin, iimax+1, 1):
                iii = ii
                for jj in range((j-ncell), (j+ncell+1), 1):
                    if jj < 0:
                        jjj = n-1 + jj
                    elif jj > n-1:
                        jjj = jj - (n-1)
                    else:
                        jjj = jj
                    xdiffx = i - ii + 0.5
                    ydiffx = j - jj

                    addx = q[iii, jjj] * xdiffx / (xdiffx**2+ydiffx**2)*DX*DY
                    Ex[i, j] += addx

                    xdiffy = i - ii
                    ydiffy = j - jj + 0.5
                    
                    addy = q[iii, jjj] * ydiffy/(xdiffy**2+ydiffy**2)*DX*DY
                    Ey[i, j] += addy
    Ex /= (2.0*np.pi)
    Ey /= (2.0*np.pi)

    return Ex, Ey

# @njit
def Efield_from_charge(q, a=10):
    DX = 1.0
    DY = 1.0
    q_shape = q.shape
    m = q_shape[0]
    n = q_shape[1]
    Ex = np.zeros(q_shape)
    Ey = np.zeros(q_shape)

    for i in range(n):
        iimin = np.max(np.array([0, i-a]))
        iimax = np.min(np.array([n, i+a]))
        # print("i=", i, "iimin=", iimin, "iimax=", iimax)
        for j in range(m):
            for ii in range(iimin, iimax+1, 1):
                iii = ii
                for jj in range((j-a), (j+a+1), 1):
                    if jj < 0:
                        jjj = m+jj
                    elif jj > m:
                        jjj = jj-m
                    else:
                        jjj = jj
                    xdiffx = i-ii+0.5
                    ydiffx = j-jj
                    addx = q[iii, jjj]*xdiffx/(xdiffx**2+ydiffx**2)*DX*DY
                    Ex[i, j] += addx
                    xdiffy = i-ii
                    ydiffy = j-jj+0.5
                    addy = q[iii, jjj]*ydiffy/(xdiffy**2+ydiffy**2)*DX*DY
                    Ey[i, j] += addy

    Ex /= (2.0*np.pi)
    Ey /= (2.0*np.pi)
    return Ex, Ey