from shock_visualization.constants import DIGITS, LSI, STEP_INIT, ION_GYROTIME, C
from visualization_package.load import load_data_from_h5file

import sys
import numpy as np
import h5py
from numba import njit
from argparse import ArgumentParser
from scipy.optimize import curve_fit
from numpy import fft

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

def parse_arguments_fourier():
    parser = ArgumentParser()

    parser.add_argument("start", type=int)
    parser.add_argument("stop", type=int)
    parser.add_argument("step", type=int)

    parser.add_argument("quantities", nargs="*")

    parser.add_argument(
        "--filetype",
        choices=["png", "pdf", "svg"],
        default="png"
    )

    parser.add_argument("--subdirectory")
    
    def str_to_tuple_float(str):
        list_of_str = str.split(",")
        list_of_int = [float(s) for s in list_of_str]
        return tuple(list_of_int)

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

def save_data_to_h5file(file_name, data, dataset_name):
    with h5py.File(file_name, "a") as f:
        f.create_dataset(dataset_name, data=data)

def func_linear(x, a, b):
    return a*x+b

def linear_fit(x, y):
    """fit y = a*x + b"""
    popt, pcov = curve_fit(func_linear, x, y)
    perr = np.sqrt(np.diag(pcov))
    a = popt[0]
    b = popt[1]
    print(f"Fitted parameters: a={a}, b={b}")
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
    file_path = f"{path_to_result}/spectra_up.h5"
    group_name = f"Step#{step}"
    group = load_data_from_h5file(file_path, group_name)
    data = group[group_name][data_name]
    return data

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

def spatial_ft(data_2d, cppu, normalize=True, hanning=True, log=True):
    """
    Computes the spatial Fourier transformation of a 2D map in k-space (k limits included)
    :param data_2d: np.ndarray. Array that will be Fourier transformed. It has to be real data
    :param cppu: float, int. Cells per Physical Unit. Number of cells that there is in the physical
    unit we want the FT in.
    Example: We want the data in Ion Skin Length (LSI), and in the data, 50 cells is one LSI. Thus, cppu=50
    :param normalize: bool. Whether the FT is normalized or not. Normalize means divide by the size of the array
    :param hanning: bool. Whether hanning is applied or not. Necessary for non-periodic data boxes
    :param log: bool. Whether we apply the logarithm to the FT for better visualisation.
    :return: ft: np.ndarray, ft_extent: list of 2 tuples
    """
    data_2d = np.array(data_2d)
    if hanning:
        data_2d *= np.sqrt(np.outer(np.hanning(data_2d.shape[0]), np.hanning(data_2d.shape[1])))

    ft = np.fft.fft2(data_2d)
    ft = np.fft.fftshift(ft)
    
    if normalize:
        ft /= data_2d.size
    ft = np.abs(ft)**2
    if log:
        ft = np.log10(ft)

    # k-space in x and y direction
    kx = fft.fftfreq(data_2d.shape[1], 1/cppu)*2*np.pi
    ky = fft.fftfreq(data_2d.shape[0], 1/cppu)*2*np.pi

    kx = fft.fftshift(kx)
    ky = fft.fftshift(ky)

    ft_extent = [np.min(kx), np.max(kx), np.min(ky), np.max(ky)]

    return ft, ft_extent, kx, ky

def simbox_area(x1,x2,y1,y2,input_unit,res_factor):
    """Are of the simulation box converted from input to resized unit"""
    x1 = int( x1 * input_unit/res_factor)
    x2 = int( x2 * input_unit/res_factor)
    y1 = int( y1 * input_unit/res_factor)
    y2 = int( y2 * input_unit/res_factor)
    return x1, x2, y1, y2

@njit
def dFdy_back(F):
    rows = F.shape[0]
    cols = F.shape[1]
    dFdy = np.zeros((rows,cols))
    for j in range(cols):
        for i in range(1,rows-1):
            dFdy[i][j] = (F[i][j] - F[i-1][j])
    return dFdy   

@njit
def dFdx_back(F):
    rows = F.shape[0]
    cols = F.shape[1]
    dFdy = np.zeros((rows,cols))
    for j in range(1,cols-1):
        for i in range(rows):
            dFdy[i][j] = (F[i][j] - F[i][j-1])
    return dFdy

@njit
def dFdy_mid(F):
    rows = F.shape[0]
    cols = F.shape[1]
    dFdy = np.zeros((rows,cols))
    for j in range(cols):
        for i in range(1,rows-1):
            dFdy[i][j] = 0.5*(F[i+1][j] - F[i-1][j])
    return dFdy   

@njit
def dFdx_mid(F):
    rows = F.shape[0]
    cols = F.shape[1]
    dFdy = np.zeros((rows,cols))
    for j in range(1,cols-1):
        for i in range(rows):
            dFdy[i][j] = 0.5*(F[i][j+1] - F[i][j-1])
    return dFdy

@njit
def curl2D3V_mid(Fx,Fy,Fz):
    curlx =  dFdy_mid(Fz)
    curly = -dFdx_mid(Fz)
    curlz =  dFdx_mid(Fy) - dFdy_mid(Fx)
    return curlx, curly, curlz

@njit
def curl2D3V_back(Fx,Fy,Fz):
    curlx =  dFdy_back(Fz)
    curly = -dFdx_back(Fz)
    curlz =  dFdx_back(Fy) - dFdy_back(Fx)
    return curlx, curly, curlz

@njit
def div2D3V_back(Fx,Fy,Fz):
    divx = dFdx_back(Fx)
    divy = dFdy_back(Fy)
    divz = 0
    div = divx + divy + divz
    return div   
