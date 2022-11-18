"""
This implementation has been taken from here, https://www.kaggle.com/code/dandm1/market-data-analysis-applying-ppca/notebook.

It is a kaggle notebook and we thank the author David, dandm1.
"""
import os

import numpy as np
from scipy.linalg import orth


class PPCA():

    def __init__(self):

        self.raw = None
        self.data = None
        self.C = None
        self.means = None
        self.stds = None
        self.eig_vals = None

    def _standardize(self, X):

        if self.means is None or self.stds is None:
            raise RuntimeError("Fit model first")

        return (X - self.means) / self.stds

    def fit(self, data, d=None, tol=1e-4, min_obs=10, verbose=False):

        self.raw = data
        self.raw[np.isinf(self.raw)] = np.max(self.raw[np.isfinite(self.raw)])

        valid_series = np.sum(~np.isnan(self.raw), axis=0) >= min_obs

        data = self.raw[:, valid_series].copy()
        N = data.shape[0]
        D = data.shape[1]

        self.means = np.nanmean(data, axis=0)
        self.stds = np.nanstd(data, axis=0)

        data = self._standardize(data)
        observed = ~np.isnan(data)
        missing = np.sum(~observed)
        data[~observed] = 0

        # initial

        if d is None:
            d = data.shape[1]

        if self.C is None:
            C = np.random.randn(D, d)
        else:
            C = self.C
        CC = np.dot(C.T, C)
        X = np.dot(np.dot(data, C), np.linalg.inv(CC))
        recon = np.dot(X, C.T)
        recon[~observed] = 0
        ss = np.sum((recon - data )**2 ) /( N *D - missing)

        v0 = np.inf
        counter = 0

        while True:

            Sx = np.linalg.inv(np.eye(d) + CC /ss)

            # e-step
            ss0 = ss
            if missing > 0:
                proj = np.dot(X, C.T)
                data[~observed] = proj[~observed]
            X = np.dot(np.dot(data, C), Sx) / ss

            # m-step
            XX = np.dot(X.T, X)
            C = np.dot(np.dot(data.T, X), np.linalg.pinv(XX + N* Sx))
            CC = np.dot(C.T, C)
            recon = np.dot(X, C.T)
            recon[~observed] = 0
            ss = (np.sum((recon - data) ** 2) + N * np.sum(CC * Sx) + missing * ss0) / (N * D)

            # calc diff for convergence
            det = np.log(np.linalg.det(Sx))
            if np.isinf(det):
                det = abs(np.linalg.slogdet(Sx)[1])
            v1 = N * (D * np.log(ss) + np.trace(Sx) - det) \
                 + np.trace(XX) - missing * np.log(ss0)
            diff = abs(v1 / v0 - 1)
            if verbose:
                print('\rAt iteration {} the diff is {:8.6f} (target {})'.format(counter, diff, tol), end='')
            if (diff < tol) and (counter > 5):
                break

            counter += 1
            v0 = v1

        C = orth(C)
        vals, vecs = np.linalg.eig(np.cov(np.dot(data, C).T))
        order = np.flipud(np.argsort(vals))
        vecs = vecs[:, order]
        vals = vals[order]

        C = np.dot(C, vecs)

        # attach objects to class
        self.C = C
        self.data = data
        self.eig_vals = vals
        self._calc_var()

    def transform(self, data=None):

        if self.C is None:
            raise RuntimeError('Fit the data model first.')
        if data is None:
            return np.dot(self.data, self.C)
        missing = np.isnan(data)
        # Obtain mean of columns as you need, nanmean is just convenient.
        it = 0
        if np.sum(missing) > 0:
            col_mean = np.nanmean(data, axis=0)
            data[missing] = np.take(col_mean, np.where(missing)[0])
            change = 1
            while (change > 1e-3):
                CC = np.dot(self.C.T, self.C)
                X = np.dot(np.dot(data, self.C), np.linalg.inv(CC))
                proj = np.dot(X, self.C.T)
                change = np.max(np.abs(data[missing] - proj[missing]))
                print('\rIteration {}. Change is {:6.4f}'.format(it, change), end='')
                data[missing] = proj[missing]
                it += 1
        return np.dot(data, self.C)

    def _calc_var(self):

        if self.data is None:
            raise RuntimeError('Fit the data model first.')

        data = self.data.T

        # variance calc
        var = np.nanvar(data, axis=1)
        total_var = var.sum()
        self.var_exp = self.eig_vals.cumsum() / total_var

    def save(self, fpath):

        np.save(fpath, self.C)

    def load(self, fpath):

        assert os.path.isfile(fpath)

        self.C = np.load(fpath)