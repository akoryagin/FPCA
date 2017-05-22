""" Functional PCA - model (Cont-Fonseca article) """

import os
import pickle
import pandas as pd
import numpy as np
from auxiliaries import *
from scipy.linalg import eig, inv
import statsmodels.api as sm


class ModelFPCA():

    def __init__(self, data, for_logs=True, for_delta=True):
        """
        model initialization

        args:
            data: dataframe with data on imp vols
                contains cols...
            for_logs: whether to use log_ivs or ivs
            for_delta: whether to perform analysis on delta_(log)_ivs or
                (log)_ivs
        """
        self.data = data
        self._for_logs = for_logs
        self._for_delta = for_delta

        self.st_date = None
        self.end_date = None

        self._integs = None
        self._basis_integs = None
        self._integs_df = None
        self._delta_integs_df = None

        self.params = None

        self._smoothed_ivs = None
        self._basis_funcs = None

    def load_integs(self, dst=None):
        """
        loader of precomputed pickled dict of integrals

        args:
            dst: path to folder with precomputed dict
        * Note that dict to be loaded depends on self._for_logs:
            If it's true, the dict with name integs_logs.pkl is loaded
            o/w, integs.pkl
                into correpsonding attribute of class-instance
        """
        filename = 'integs_logs.pkl' if self._for_logs else 'integs.pkl'
        filepath = os.path.join(dst, filename) if dst is not None else filename

        # load dict with h_i (log)sigma_t integrals
        with open(filepath, 'rb') as file:
            self._integs = pickle.load(file)

        # load ndarray with integrals of basis funcs
        filepath = os.path.join(
            dst, 'basis.pkl') if dst is not None else 'basis.pkl'
        with open(filepath, 'rb') as file:
            self._basis_integs = pickle.load(file)

        # transform dicts in dfs
        self._integs_df = pd.DataFrame.from_dict(
            self._integs, orient='index').sort_index()
        self._delta_integs_df = self._integs_df.diff().dropna()

    def set_params(self, num_mon=3, num_mat=3, epsilon=5.5):
        """
        set params of the model
        params are
            epsilon of smoothing
            num_mat: number of nodes in mat-dimension
            num_mon: number of nodes in mon-dimension
        """
        self.num_mat = num_mat
        self.num_mon = num_mon
        self.epsilon = epsilon

    def compute_smoothers_and_basis(self):
        """
        add dict _smoothed_ivs with smoothers
            to self._all_dates
        """
        # compute smoothers
        self._smoothed_ivs = {}
        for dat in self._all_dates:
            self._smoothed_ivs[dat] = smoothed_IVS(self.data, dat,
                                                   epsilon=self.epsilon)
        # add basis funcs to self
        self._basis_funcs = []
        grid_basis = get_grid_basis(num_mon=self.num_mon,
                                    num_mat=self.num_mat)
        for i in range(len(grid_basis)):
            hi = get_basis_func(grid_basis[i, 0], grid_basis[
                                i, 1], epsilon=0.3)
            self._basis_funcs.append(hi)

    @property
    def _mean_ivs(self):
        """
        mean function of ivs inside model-range
        applicable only for non-delta models
        """
        needed_ivs = [self._smoothed_ivs[dat] for dat in self._frame_dates]
        if self._for_logs:
            def mean(tau, m):
                needed_vals = [np.log(ivs(tau, m)) for ivs in needed_ivs]
                return np.nanmean(needed_vals)
            return mean
        else:
            def mean(tau, m):
                needed_vals = [ivs(tau, m) for ivs in needed_ivs]
                return np.nanmean(needed_vals)
            return mean

    @property
    def _all_dates(self):
        return sorted(self._integs.keys())

    @property
    def _frame_dates(self):
        dts = [dat for dat in self._all_dates if dat >=
               self.st_date and dat <= self.end_date]
        return sorted(dts)[1:]

    def set_range(self, from_start=True, days_num=100, st_date=None, end_date=None):
        all_dts = self._all_dates
        if from_start:
            self.st_date = all_dts[0]
            self.end_date = all_dts[days_num - 1]
        else:
            self.st_date = st_date
            self.end_date = end_date

    def move_window(self, step=1):
        all_dts = self._all_dates
        st_pos = all_dts.index(self.st_date)
        end_pos = all_dts.index(self.end_date)

        new_st = all_dts[st_pos + step]
        new_end = all_dts[end_pos + step]

        self.set_range(st_date=new_st, end_date=new_end,
                       from_start=False)

    @property
    def _matrix_C(self):
        if self._for_delta:
            df = self._delta_integs_df
        else:
            df = self._integs_df

        arr_integs = df.loc[self._frame_dates, :].sort_index().values
        term_1 = np.dot(arr_integs.transpose(), arr_integs) / \
            len(self._frame_dates)
        means = arr_integs.mean(axis=0).reshape(-1, 1)
        term_2 = np.dot(means, means.transpose())

        # print(term_1.shape, term_2.shape)
        return (term_1 - term_2)

    @property
    def _matrix_B(self):
        return self._basis_integs

    @property
    def _matrix_A(self):
        B_inv = inv(self._matrix_B)
        G = np.dot(B_inv, self._matrix_C)
        decomp = eig(G)

        X = decomp[1]
        A = X.transpose()
        return A

    @property
    def _diagonal_D(self):
        B_inv = inv(self._matrix_B)
        G = np.dot(B_inv, self._matrix_C)
        decomp = eig(G)

        D = decomp[0]
        return D

    @property
    def pc_series(self):
        """
        Note that non-normalized projections are computed here
        That is, non-normalized eigmodes are taken and
        sigmas are projected on them
        """
        if self._for_delta:
            df = self._delta_integs_df
        else:
            df = self._integs_df

        arr_integs = df.loc[self._frame_dates, :].sort_index().values

        # if not for._delta project demeaned (log) sigma on eigenmodes
        # equivalent to demean projections
        if not self._for_delta:
            arr_integs -= arr_integs.mean(axis=0)

        arr_pcs = np.dot(arr_integs, self._matrix_A.transpose())

        return pd.DataFrame(index=self._frame_dates, data=arr_pcs,
                            columns=['pc_' + str(i) for i in range(arr_pcs.shape[1])])

    def _get_basis_poly(self, coeffs):
        coeffs = np.squeeze(np.asarray(coeffs))

        if len(coeffs) != len(self._basis_funcs):
            raise ValueError(
                'Number of coeffs is not equal to numbers of basis funcs')

        def poly(tau, m):
            res = 0
            for k in range(len(self._basis_funcs)):
                hk = self._basis_funcs[k]
                res += coeffs[k] * hk(tau, m)
            return res
        return poly

    def _compute_basis_poly(self, coeffs, pts):
        """
        compute values of polynom of basis funcs in pts
        args:
            pts: array [.. X 2] of different tau, m
            coeffs: coeffs of poly-representation
        """
        coeffs = np.squeeze(np.asarray(coeffs)).reshape(1, -1)
        pts = np.asarray(pts).reshape(-1, 2)

        if coeffs.shape[1] != len(self._basis_funcs):
            raise ValueError(
                'Number of coeffs is not equal to numbers of basis funcs')

        # caclulate vals of basis funcs in pts of interest
        basis_vals = np.zeros(shape=(len(self._basis_funcs), len(pts)))

        for i in range(basis_vals.shape[0]):
            for j in range(basis_vals.shape[1]):
                tau, m = pts[j, 0], pts[j, 1]
                basis_vals[i, j] = self._basis_funcs[i](tau, m)

        return np.squeeze(np.dot(coeffs, basis_vals))

    def _compute_eigmodes_poly(self, coeffs, pts):
        """
        compute values of polynom of eigenmodes in pts
            with coeffs
        args:
            pts: array [.. X 2] of different tau, m
            coeffs: coeffs of poly-representation
        """
        A = self._matrix_A
        if len(A) != len(np.squeeze(np.asarray(coeffs))):
            raise ValueError('coeffs-number should be equal to number of pcs')

        coeffs = np.squeeze(np.asarray(coeffs)).reshape(1, -1)
        basis_coeffs = np.dot(coeffs, A)
        return self._compute_basis_poly(basis_coeffs, pts)

    def _compute_mean(self, pts):
        """
        compute vals of mean-function in pts
        only for non-delta models
        """
        pts = np.asarray(pts).reshape(-1, 2)
        needed_ivs = [self._smoothed_ivs[dat] for dat in self._frame_dates]

        vals = np.zeros(shape=(len(needed_ivs), len(pts)))
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                vals[i, j] = needed_ivs[i](pts[j, 0], pts[j, 1])

        return vals.mean(axis=0)

    @property
    def _eigennorms(self):
        """
        calculate squared-norms of eigenmodes
        """
        return np.dot(np.dot(self._matrix_A, self._matrix_B),
                      self._matrix_A.transpose()).diagonal()

    @property
    def pc_series_sq_normalized(self):
        """
        normalize series of pcs on norms of eigenmodes
        """
        return self.pc_series / self._eigennorms

    def _get_update(self, pc_coeffs, n_comps=2):
        """
        returns update based on array of pc_coeffs
            n_comps also supplemented

        pc_coeffs are assumed to be squared normalized


        """
        pc_coeffs = np.squeeze(np.asarray(pc_coeffs)
                               ).reshape(-1)[:n_comps].reshape(1, n_comps)
        basis_coeffs = np.dot(pc_coeffs, self._matrix_A[:n_comps, ])
        #  print(basis_coeffs.shape)
        return self._get_basis_poly(basis_coeffs)

    @staticmethod
    def _eval_fit(fit, p_fit, metric='mape'):
        """
        evaluate fit quality of fit against p_fit (perfect fit)
        metric can be either
            'mape' - mean absolite percentage error
            'mae' - mean absolute error
            'mse' - mean squared error
            'rmse' - root mean squared error
        """
        p_fit, fit = np.asarray(p_fit), np.asarray(fit)
        devs = np.abs(p_fit - fit)
        if metric == 'mape':
            return np.mean(devs / np.abs(p_fit))
        elif metric == 'mae':
            return np.mean(devs)
        elif metric == 'mse':
            return np.mean(np.square(devs))
        elif metric == 'rmse':
            return np.sqrt(np.mean(np.square(devs)))

    def assess_fit_quality(self, day, fit, with_smoothed=False,
                           metric='mape'):
        """
        assess quality of fit given
        args:
            day: for which day to asess the quality
            fit: callable, ivs to compare
                note that it's actually log-ivs if
                the model is for logs
            with_smoothed: if True, fit is compared to
                smoothed ivs rather than real
                note that smoothers are simple ivs,
                not logs
            metric: evaluation metric. Can be either
                mape, mae, rmse, mse, or a list of
                any of those
        """
        pts = self.data[self.data.date == day][
            ['maturity', 'moneyness', 'imp_vol']].values

        # ground-truth function
        # depends o the number of pts for a specific day
        def perfect_fit(i):
            tau, m = pts[i, 0], pts[i, 1]
            if with_smoothed:
                if self._for_logs:
                    return np.log(self._smoothed_ivs[day](tau, m))
                else:
                    return self._smoothed_ivs[day](tau, m)
            else:
                if self._for_logs:
                    return np.log(pts[i, 2])
                else:
                    return pts[i, 2]

        p_fits = []
        fits = []
        for i in range(len(pts)):
            tau, m = pts[i, 0], pts[i, 1]
            p_fits.append(perfect_fit(i))
            fits.append(fit(tau, m))

        fits = np.asarray(fits)
        p_fits = np.asarray(p_fits)

        if isinstance(metric, str):
            return self._eval_fit(fits, p_fits, metric)

        elif isinstance(metric, list):
            res = [self._eval_fit(fits, p_fits, metr) for metr in metric]
            return res
    
    def in_sample_fit(self, day, n_comps=2):
        """
        return in-sample fit given day inside model range
        """
        if day not in self._frame_dates:
            raise ValueError('day outside of model range')

        """
        calculate pc-based adjustment to fit of ivs
            will be added to either mean-ivs or lagged-ivs
        """
        pcs_sq_norm = self.pc_series_sq_normalized.loc[day, :]
        pcs_sq_norm = np.asarray(pcs_sq_norm)[:n_comps]
        update = self._get_update(pcs_sq_norm, n_comps)

        if not self._for_delta:
            update = self._get_update(pcs_sq_norm, n_comps)
            mean = self._mean_ivs
            # if not for delta
            # then mean + update based on sq normalized pcs

            def fit(tau, m):
                return mean(tau, m) + update(tau, m)
            return fit

        elif self._for_delta:
            # if for delta
            # then smoothed ivs(lagged date) + update based on sq norm pcs
            pos_lagged = self._all_dates.index(day) - 1
            dat_lagged = self._all_dates[pos_lagged]
            ivs_lagged = self._smoothed_ivs[dat_lagged]
            
            def fit(tau, m):
                if self._for_logs:
                    return np.log(ivs_lagged(tau, m)) + update(tau, m)
                else:
                    return ivs_lagged(tau, m) + update(tau, m)
            return fit

    def day_ahead_fit(self, n_comps=2, var_order=2):
        """
        return day-ahead out-of-sample
            forecast of IVS
        forecast is build for next day after model range

        args:
            n_comps: number of component to forecast and use
            in building update
        """

        # build var-forecast of pcs for next day
        columns = ['pc_' + str(i) for i in range(n_comps)]
        pcs_needed = self.pc_series_sq_normalized[columns]
        var_model = sm.tsa.VAR(pcs_needed)

        # fit var
        fitted = var_model.fit(var_order)

        # day-ahead forecast of pcs
        pcs_forec = np.squeeze(fitted.forecast(
            pcs_needed.values[-var_order:], 1)).reshape(-1)

        # now obtain update based on forecasted pcs
        # and add it to mean/lagged ivs
        update = self._get_update(pcs_forec, n_comps)
        if not self._for_delta:
            mean = self._mean_ivs
            # if not for delta
            # then mean + update based on sq normalized pcs

            def fit(tau, m):
                return mean(tau, m) + update(tau, m)
            return fit

        elif self._for_delta:
            # if for delta
            # then smoothed ivs(lagged date) + update based on sq norm pcs
            # take last day of model-range
            day = self._frame_dates[-1]
            # pos_lagged = self._all_dates.index(day) - 1
            # dat_lagged = self._all_dates[pos_lagged]
            ivs_lagged = self._smoothed_ivs[day]

            def fit(tau, m):
                if self._for_logs:
                    return np.log(ivs_lagged(tau, m)) + update(tau, m)
                else:
                    return ivs_lagged(tau, m) + update(tau, m)
            return fit

    def get_ith_eigenmode(self, i):
        """
        get i-th eigenmode comoputed using model range
        args:
            i - number of eigenmode
        returns callable (function object)
        """
        eig_coeffs = self._matrix_A[i, :]
        return self._get_basis_poly(eig_coeffs)

    def jth_ith(self, i, j):
        eig_modei = self.get_ith_eigenmode(i)
        eig_modej = self.get_ith_eigenmode(j)

        res = dblquad(lambda m, tau: eig_modej(
            tau, m) * eig_modei(tau, m), mat_lb, mat_ub, mon_l_bound, mon_u_bound)[0]
        return res

    def check_precomputed(self):
        """
        check that all needed dicts are loaded
        compute delta-dict  if necessary 
        """

        if self._integs is None:
            raise ValueError('the dict with integs is not loaded')

    def get_proj_ith(self, date, i):
        dic_delta_integs = self._delta_integs_df
        A = self._matrix_A

        proj = 0
        for k in range(A.shape[0]):
            proj += A[i, k] * dic_delta_integs.loc[date][k]

        return proj
