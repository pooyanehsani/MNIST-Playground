import numpy as np

from scipy.optimize import (
    check_grad,
    fmin_cg,
    fmin_ncg,
    fmin_bfgs,
)

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
def euclid_dis(x1, x2):
    return (np.sum(x1 * x1, 1)[:, np.newaxis] +np.sum(x2 * x2, 1)[np.newaxis, :] -np.dot(x1, (2 * x2.T)))

def cost_nca(A,x,y):
    N, D = x.shape
    K = A.shape[0]
    #cost function
    cf = np.dot(A, x.T)
    sd = euclid_dis(cf.T, cf.T)
    np.fill_diagonal(sd, np.inf)
    mm = np.min(sd, axis=0)
    kk = np.exp(mm - sd)
    np.fill_diagonal(kk, 0)
    Z_p = np.sum(kk, 0)
    p_mn = kk / Z_p[np.newaxis, :]
    mask = y[:, np.newaxis] == y[np.newaxis, :]
    p_n = np.sum(p_mn * mask, 0)
    ff = - np.sum(p_n)

    # Back-propagate gradient
    kk_bar = - (mask - p_n[np.newaxis, :]) / Z_p[np.newaxis, :]
    ee_bar = kk * kk_bar
    zz_bar_part = ee_bar + ee_bar.T
    zz_bar = 2 * (np.dot(cf, zz_bar_part) - (cf * np.sum(zz_bar_part, 0)))
    gg = np.dot(zz_bar, x)
    return ff, gg

class NCA(BaseEstimator, TransformerMixin):
    def __init__(self, k):
        self.K = k
        self._fit = self._fit_gradient_descent

    def fit(self, X, y):
        N, D = X.shape
        self.A = np.random.randn(self.K, D) / np.sqrt(N)
        return self._fit(X,y)
    def _fit_gradient_descent(self, X, y):
        self.learning_rate = 0.001
        self.error_tol = 0.001
        self.max_iter = 400
        curr_error = None
        for iter in range(self.max_iter):
            f, g = cost_nca(self.A, X, y)
            self.A -= self.learning_rate * g
            prev_error = curr_error
            curr_error = f
            if prev_error and np.abs(curr_error - prev_error) < self.error_tol:
                break
        return self
    def transform(self,X):
        return np.dot(X, self.A.T)
    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)
if __name__ == "__main__":
	NCA()


