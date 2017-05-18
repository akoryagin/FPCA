import numpy as np
import datetime as dt
from scipy.interpolate import Rbf
from scipy.integrate import dblquad, nquad
import itertools
import os
import pickle


"""
bounds of rectangular shape in mat X mon - plane
"""
mon_lb = 0.9
mon_ub = 1.14

mat_lb = 0.082
mat_ub = 1.0


"""
limits of integration are defined here
"""

coeffs = {'intercept_lower': 0.9438160044900755,
          'intercept_upper': 1.0391042936788095,
          'slope_lower': -0.0788207041931871,
          'slope_upper': 0.16085419519136232}

mon_l_bound = lambda tau: coeffs[
    'intercept_lower'] + coeffs['slope_lower'] * tau
mon_u_bound = lambda tau: coeffs[
    'intercept_upper'] + coeffs['slope_upper'] * tau


def smoothed_IVS(data, date, mute=True, epsilon=5.5):
    """
    returns ivs smoothing for a date as a
        callable object, can be used as a function
    args:
        data must be a dataframe with cols
            date
            maturity 
            moneyness
            imp_vols
        date is a day for which smoothing has to be
            calculated
        epsilon sets gaussian kernel diam
    """
    tmp = data[data.date == date]
    if len(tmp) == 0:
        if not mute:
            print('sorry, only wednesdays are in the dataset')
        return

    X = tmp.maturity.values
    Y = tmp.moneyness.values
    Z = tmp.imp_vol.values
    smoothed = Rbf(X, Y, Z, epsilon=epsilon,
                   smooth=0.00000001, function='gaussian')
    return smoothed


def log_smoothed_IVS(data, date, mute=True, epsilon=5.5):
    """
    returns log ivs smoothing for a date as a
        callable object, can be used as a function
    args:
        data must be a dataframe with cols
            date
            maturity 
            moneyness
            imp_vols
        date is a day for which smoothing has to be
            calculated
        epsilon sets gaussian kernel diam
    """
    tmp = data[data.date == date]
    if len(tmp) == 0:
        if not mute:
            print('sorry, only wednesdays are in the dataset')
        return None

    X = tmp.maturity.values
    Y = tmp.moneyness.values
    Z = tmp.imp_vol.values
    smoothed = Rbf(X, Y, Z, epsilon=epsilon,
                   smooth=0.00000001, function='gaussian')

    def log_smoothed(tau, m):
        return np.log(smoothed(tau, m))

    return log_smoothed


def get_basis_func(tau_node, m_node, epsilon=0.3):
    """
    return basis gaussian with node in tau_node, m_node
        epsilons sets the radius
    if tau_node == -1
        return const = 1 function
    args:
        tau_node, m_node: node in maturity X moneyness plane
        espilon: radius of kernel
    """
    if tau_node == -1:
        def func(tau, m):
            return 1
        return func
    else:
        def func(tau, m):
            x = (tau - mat_lb) / (mat_ub - mat_lb)
            y = (m - mon_lb) / (mon_ub - mon_lb)
            x_node = (tau_node - mat_lb) / (mat_ub - mat_lb)
            y_node = (m_node - mon_lb) / (mon_ub - mon_lb)
            return 1 / epsilon * np.exp(1 / epsilon * (-(x - x_node)**2 - (y - y_node)**2))
        return func


def get_grid_basis(num_mon=3, num_mat=3):
    """
    construct set of nodes for basis gaussians
        return ndarray with shape [set_size X 2] of (tau_node, m_node)
        scattered in mat X mon plane

    set_size = num_mon * num_mat + 1 (9 nodes and const-func)
    args:
        num_mat: number of nodes in mat-dimension
        num_mon: number of nodes in mon-dimension
    """
    grid_basis_mat = np.linspace(mat_lb, mat_ub, num_mat)
    grid_basis = np.zeros(shape=[num_mon * num_mat, 2])

    for i in range(len(grid_basis_mat)):
        tau = grid_basis_mat[i]
        lb = mon_l_bound(tau)
        ub = mon_u_bound(tau)
        grid_basis_mon = np.linspace(lb, ub, num_mon)

        for j in range(len(grid_basis_mon)):
            mon = grid_basis_mon[j]
            grid_basis[i * num_mon + j][0] = tau
            grid_basis[i * num_mon + j][1] = mon

    grid_basis = np.vstack((grid_basis, [-1, 1]))
    return grid_basis


def integ_sigt_hi(data, date_t, node, for_logs=True, clip_level=0.01):
    """
    compute integral of (log) sigma_t h_i over trapezoid
    return the value of integral
    args:
        data:   must be a dataframe with cols
                date
                maturity 
                moneyness
                imp_vols
        date_t: day for which smoothed-IVS (log) has to be
                taken 
        node:   tuple (tau_node, m_node)
                gaussian with center in node is h_i
        for_logs: whether to calculate integral for IVS or log-IVS
                  default option: for_logs=True
                  assume that analysis by default is performed
                  for logs
    """
    IVS_t = smoothed_IVS(data, date_t)
    hi = get_basis_func(*node, epsilon=0.3)

    if IVS_t:
        if for_logs:
            return dblquad(lambda m, tau: np.log(max(IVS_t(tau, m), clip_level)) * hi(tau, m),
                           mat_lb, mat_ub, mon_l_bound, mon_u_bound)[0]
        else:
            return dblquad(lambda m, tau: IVS_t(tau, m) * hi(tau, m),
                           mat_lb, mat_ub, mon_l_bound, mon_u_bound)[0]


def compute_all_integs(data, grid_basis, dts=None, dump=False, for_logs=True, verbose=False):
    """
    compute dictionary I(i, t), return it
    args:
        data: must be a dataframe with cols
            date
            maturity
            moneyness
            imp_vols
        grid_basis: array of nodes for basis funcs
        dts: range of dates for which the dict is to be calculated
            if None, all possible dates from data are used
        dump: whether to dump resulting dictionary
            by default is not dumped
        for_logs: whether to compute integs for logs or simple IVS
            default option is to use logs
        verbose: if True, print the progress
    """
    if dts is None:
        dts = data.date.unique()

    dic_integs = {}
    ctr = 0
    for dat in dts:
        dic_integs[dat] = []
        ctr += 1
        if ctr % 20 == 0 and verbose:
            print(ctr)
        for i in range(grid_basis.shape[0]):
            node = (grid_basis[i, 0], grid_basis[i, 1])
            dic_integs[dat].append(integ_sigt_hi(
                data=data, date_t=dat, node=node, for_logs=for_logs))

    if dump:
        if for_logs:
            filename = 'integs_logs.pkl'
        else:
            filename = 'integs.pkl'
        if not os.path.exists(filename):
            with open(filename, 'wb') as file:
                pickle.dump(dic_integs, file)

    return dic_integs


def compute_matrix_B(grid_basis, dump=False):
    """
    compute matrix B with integrals of
        multiplications of basis functions
    args:
        grid_basis: array of nodes for basis funcs
    """
    B = np.identity(grid_basis.shape[0])

    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            hi = get_basis_func(grid_basis[i, 0], grid_basis[
                                i, 1], epsilon=0.3)
            hj = get_basis_func(grid_basis[j, 0], grid_basis[
                                j, 1], epsilon=0.3)

            B[i, j] = dblquad(lambda m, tau: hi(
                tau, m) * hj(tau, m), mat_lb, mat_ub, mon_l_bound, mon_u_bound)[0]
    if dump:
        filename = 'basis.pkl'
        if not os.path.exists(filename):
            with open(filename, 'wb') as file:
                pickle.dump(B, file)

    return B
