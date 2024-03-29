
import scipy.special

import PatternLib.probability as pr
from PatternLib.blueprint import *
import numpy as np
from PatternLib.probability import GAU_ND_logpdf
from PatternLib.preproc import get_cov, filter_outliers
from scipy.optimize import fmin_l_bfgs_b
from Examples.filters import filters

class Perceptron(Faucet):

    def __init__(self, iterations=100, alpha=0.1, seed=None):
        self.Y = None
        self.X = None
        self.labels = None
        self.iter = iterations
        self.alpha = alpha
        self.weights = None
        self.seed = seed

    def fit(self, x, y):
        if self.seed is not None:
            np.random.seed(self.seed)

        rand = np.random.choice(x.shape[0], x.shape[0])
        self.X = np.hstack([np.ones((x.shape[0], 1)), x])[rand, :].T
        self.Y = y.reshape((-1, 1))[rand, :]
        self.labels = np.unique(y)

        def single(y_true):
            w = np.random.random((self.X.shape[0], 1)).T - 0.5
            for i in range(self.iter):
                for data, y_tr in zip(self.X.T, y_true.T):
                    data = data.reshape((1, -1))
                    y_predict = np.heaviside(w @ data.T, 0)
                    w = w + self.alpha * (y_tr - y_predict) * data
            return w

        lis = []
        for label in self.labels:
            y_l = (self.Y == label)
            lis.append(single(y_l.T))
        self.weights = np.vstack(lis).T
        return self

    def predict(self, x, return_prob=False):
        one = np.ones((x.shape[0], 1))
        x = np.hstack([one, x]).T
        y = self.weights.T @ x
        if return_prob:
            return y
        return self.labels[np.argmax(y, axis=0)]

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)


class GaussianClassifier(Faucet):

    def __init__(self, outlier_filter=False):
        self.estimates = []
        self.labels = None
        self.Spost = None
        self.outlier_filter = outlier_filter

    def fit(self, x, y):
        if self.outlier_filter:
            mask = filter_outliers(x, y, filters)
            x = x[mask]
            y = y[mask]
        self.estimates = []
        un, con = np.unique(y, return_counts=True)
        for label, count in zip(un, con):
            matrix = x[y == label, :]
            self.estimates.append((label, np.mean(matrix, 0), get_cov(matrix), count / x.shape[0]))
        return self

    def predict(self, x, return_prob=False):
        scores = []
        ll = []
        for label, mu, cov, prob in self.estimates:
            scores.append(GAU_ND_logpdf(x.T, mu.reshape(-1, 1), cov) + np.log(prob))
            ll.append(GAU_ND_logpdf(x.T, mu.reshape(-1, 1), cov))
        SJoint = np.hstack([value.reshape(-1, 1) for value in scores])
        logsum = scipy.special.logsumexp(SJoint, axis=1)
        self.Spost = SJoint - logsum.reshape(1, -1).T
        res = np.argmax(self.Spost, axis=1)
        return res if not return_prob else res, ll[1] - ll[0]

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)

    def __str__(self):
        return "Gaussian()"



class NaiveBayes(GaussianClassifier):

    def fit(self, x, y):
        un, con = np.unique(y, return_counts=True)
        for label, count in zip(un, con):
            matrix = x[y == label, :]
            cov = np.diag(np.var(matrix, 0))
            self.estimates.append((label, np.mean(matrix, 0), cov, count / x.shape[0]))
        return self

    def __str__(self):
        return "NaiveBayes()"


class TiedGaussian(GaussianClassifier):

    def fit(self, x, y):
        super().fit(x, y)
        sigma = (1 / y.shape[0]) * sum([sigma * sum(y == label) for label, mu, sigma, prob in self.estimates])
        self.estimates = [(label, mu, sigma, prob) for label, mu, _, prob in self.estimates]
        return self

    def __str__(self):
        return "TiedGaussian()"


class MultiNomial(Faucet):

    def __init__(self, eps=0.01):
        self.eps = eps
        self.labels = None
        self.estimate = None
        self.Spost = None

    def fit(self, x, y):
        self.labels = np.unique(y)
        arr_list = []
        for label in self.labels:
            arr = x[y == label, :].sum(axis=0) + self.eps
            arr = np.log(arr) - np.log(arr.sum())
            arr_list.append(arr)
        self.estimate = np.vstack(arr_list)

    def predict(self, x, return_prob=False):
        sjoint = x @ self.estimate.T
        logsum = scipy.special.logsumexp(sjoint, axis=1)
        self.Spost = sjoint - logsum.reshape(-1, 1)
        res = self.labels[np.argmax(x @ self.estimate.T, axis=1)]
        return res if not return_prob else res, np.exp(self.Spost)

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)


class LogisticRegression(Faucet):

    def __init__(self, norm_coeff=1, prior=None, max_fun=15000, max_iter=15000):
        self.norm_coeff = norm_coeff
        self.max_fun = max_fun
        self.max_iter = max_iter
        self.w = None
        self.b = None
        self.binary = None
        self.prior = prior

    def fit(self, x, y):
        if np.unique(y).shape[0] == 2:
            self.fit_2class(x, y)
            self.binary = True
        else:
            self.fit_nclass(x, y)
            self.binary = False

    def fit_2class(self, x, y):
        def objective_function_reg(arr):
            w, b = arr[:-1].reshape(-1, 1), arr[-1]
            regular = self.norm_coeff / 2 * np.power(np.linalg.norm(w.T, 2), 2)
            s = w.T @ x.T + b
            body1 = self.prior / (y == 1).sum() * np.log1p(np.exp(-s[:, y == 1]))
            body0 = (1 - self.prior) / (y == 0).sum() * np.log1p(np.exp(s[:, y == 0]))
            return regular + np.sum(body0) + np.sum(body1)

        def objective_function(arr):
            w, b = arr[:-1].reshape(-1, 1), arr[-1]
            regular = self.norm_coeff / 2 * np.power(np.linalg.norm(w.T, 2), 2)
            s = w.T @ x.T + b
            body = y * np.log1p(np.exp(-s)) + (1 - y) * np.log1p(np.exp(s))
            return regular + np.sum(body) / y.shape[0]

        m, f, d = fmin_l_bfgs_b(objective_function if self.prior is None else objective_function_reg,
                                np.zeros(x.shape[1] + 1), approx_grad=True,
                                maxfun=self.max_fun, maxiter=self.max_iter)
        self.w = m[:-1]
        self.b = m[-1]

    def fit_nclass(self, x, y):
        T = np.array((y.reshape(-1, 1) == np.unique(y)), dtype="int32")
        sh = (np.unique(y).shape[0], x.shape[1] + 1)

        def objective_function(arr):
            arr = arr.reshape(sh)
            w, b = arr[:, :-1], arr[:, -1].reshape(-1, 1)
            regular = self.norm_coeff / 2 * np.sum(w * w)
            s = w @ x.T
            ls = scipy.special.logsumexp(s + b)
            ly = s + b - ls
            return regular - np.sum(T * ly.T) / x.shape[0]

        init_w = np.zeros(sh)
        m, f, d = fmin_l_bfgs_b(objective_function, init_w,
                                approx_grad=True,
                                maxfun=self.max_fun,
                                maxiter=self.max_iter)
        m = m.reshape(sh)
        self.w = m[:, :-1]
        self.b = m[:, -1].reshape(-1, 1)

    def predict(self, x, return_prob=False):
        if self.w is None:
            raise NoFitError()
        if self.binary:
            sc = self.w @ x.T + self.b
            return sc > 0, sc if return_prob else sc > 0
        else:
            sc = self.w @ x.T + self.b
            return np.argmax(sc, axis=0), sc if return_prob else np.argmax(sc, axis=0)

    def fit_predict(self, x, y):
        return self.predict(self.fit(x, y))

    def __str__(self):
        return f"LogisticRegression(norm={self.norm_coeff})"


class SVM(Faucet):

    def __init__(self, k=10, c=10, ker=None, prior=None, paramker=None):
        self.C = c
        self.K = k
        self.W = None
        self.b = None
        self.w_cap = None
        self.dual_gap = None
        self.primal = None
        self.dual = None
        self.ker = ker
        self.paramker = paramker
        self.z = None
        self.x = None
        self.alpha = None
        self.prior = prior

    def fit(self, x, y):

        self.x = x.T
        DTR = x.T
        z = np.where(y == 1, 1, -1)
        self.z = z
        DTRc = np.row_stack((DTR, self.K * np.ones(DTR.shape[1])))

        x0 = np.zeros(DTR.shape[1])
        if self.prior is None:
            bounds = [(0, self.C) for _ in range(DTR.shape[1])]
        else:
            prior_emp = (1.0*(self.z > 0)).sum()/y.size
            CT, CF = self.C*self.prior/prior_emp, self.C*(1-self.prior)/(1-prior_emp)
            bounds = [(0, CF) if y[i] == 0 else (0, CT) for i in range(x.shape[0])]
        if self.ker is None:
            G = self.kernel(DTRc, DTRc)
        else:
            G = self.kernel(DTR, DTR)
        H = G * z.reshape(z.shape[0], 1)
        H = H * z

        def SVM_dual_kernel_obj(v):
            a = v
            J1 = a.T @ H @ a / 2
            J2 = - a.T @ np.ones(a.shape)
            grad = H @ a - np.ones(a.shape)
            return J1 + J2, grad.reshape(DTRc.shape[1])

        m, self.primal, _ = fmin_l_bfgs_b(SVM_dual_kernel_obj, x0, bounds=bounds)
        self.alpha = m
        res = np.sum(m * z.T * DTRc, axis=1)
        self.W = res[:-1]
        self.b = res[-1]
        self.w_cap = res

    def kernel(self, xi, xj):
        if self.ker is None:
            return xi.T @ xj
        if self.ker == "Poly":
            return np.power(xi.T @ xj + self.paramker[1], self.paramker[0]) + self.K
        if self.ker == "Radial":
            a = np.repeat(xi, xj.shape[1], axis=1)
            b = np.tile(xj, xi.shape[1])
            m = (np.linalg.norm(a - b, axis=0) ** 2).reshape((xi.shape[1], xj.shape[1]))
            return np.exp(-self.paramker[0] * m) + self.K

    def predict(self, x, return_prob=False):
        if self.ker is None:
            score = self.W.T @ x.T + self.b * self.K
        else:
            score = (self.alpha * self.z) @ self.kernel(self.x, x.T)
        return np.where(score > 0, 1, 0), score if return_prob else np.where(score > 0, 1, 0)

    def fit_predict(self, x, y):
        pass

    def __str__(self):
        if self.ker is None:
            return f"SVM(kernel=Linear, C={self.C}, regularization={self.K})"
        if self.ker == "Poly":
            return f"SVM(kernel=Polynomial, C={self.C}, regularization={self.K}, d={self.paramker[0]}, c={self.paramker[1]})"
        if self.ker == "Radial":
            return f"SVM(kernel=Radial, C={self.C}, regularization={self.K},  gamma={self.paramker[0]})"
        return "Error"


class GaussianMixture(Faucet):

    def __init__(self, alpha=0.1, N=2, tied=False, diag=False, psi=None):
        self.alpha = alpha
        self.N = N
        self.tied = tied
        self.psi = psi
        self.gmm_est = {}
        self.diag = diag

    def fit(self, x, y):
        self.gmm_est = {}
        for label in np.unique(y):
            elem = x[y == label, :].T
            post = elem.shape[1] / x.shape[0]
            estimate = pr.LGB_estimation(elem, posterior=post, alpha=self.alpha, n=self.N,
                                         psi=self.psi, tied=self.tied, diag=self.diag)
            self.gmm_est[label] = estimate

    def predict(self, x, return_prob=False):
        res = []
        for label in self.gmm_est:
            res.append(pr.logpdf_GMM(x.T, self.gmm_est[label]))
        log_matr = np.vstack(res)
        if return_prob:
            return np.argmax(log_matr, axis=0), log_matr[1, :] - log_matr[0, :]
        return np.argmax(log_matr, axis=0)

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x, y)

    def __str__(self):
        res = ""
        if self.tied and self.diag:
            res += f"TiedDiagGMM(alpha={self.alpha}"
        elif self.tied:
            res += f"TiedCovarianceGMM(alpha={self.alpha}"
        elif self.diag:
            res += f"DiagonalGMM(alpha={self.alpha}"
        else:
            res += f"FullGMM(alpha={self.alpha}"
        res += f", N={2**self.N}"
        if self.psi is not None:
            res += f", psi={self.psi})"
        return res
