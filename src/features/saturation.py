"""
saturation.py
=============
Hill-curve saturation as an sklearn-compatible transformer.

    sat(a) = a^s / (a^s + k^s)

k : half-saturation point (adstocked-spend level at which the channel hits
    50% of its own max response). Scaled per-channel so the search space
    auto-adapts to each channel's spend range.
s : steepness (higher = sharper knee into diminishing returns).

Output is left UNSCALED by a ceiling -- the ceiling/magnitude is absorbed
by the downstream ElasticNet coefficient, so this transformer only needs to
supply the correct SHAPE of diminishing returns, not the correct scale.
This keeps the transformer stateless and the total parameter count lower
for the joint hyperparameter search.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HillSaturation(BaseEstimator, TransformerMixin):
    """
    Applies per-column Hill saturation.

    Parameters
    ----------
    k : dict[int, float] or float
        Half-saturation point per column (column index -> value) or scalar.
    s : dict[int, float] or float
        Steepness per column or scalar.
    """

    def __init__(self, k=100000.0, s=1.5):
        self.k = k
        self.s = s

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n_rows, n_cols = X.shape
        X = np.clip(X, 0, None)

        k_arr = self._to_array(self.k, n_cols, default=100000.0)
        s_arr = self._to_array(self.s, n_cols, default=1.5)
        k_arr = np.where(k_arr <= 0, 1.0, k_arr)  # guard against zero/negative k

        with np.errstate(invalid="ignore", divide="ignore"):
            numer = X ** s_arr
            denom = numer + k_arr ** s_arr
            out = np.divide(numer, denom, out=np.zeros_like(X), where=denom > 0)
        return out

    @staticmethod
    def _to_array(param, n_cols, default):
        if isinstance(param, dict):
            return np.array([param.get(i, default) for i in range(n_cols)])
        elif np.isscalar(param):
            return np.full(n_cols, param)
        arr = np.asarray(param, dtype=float)
        if arr.shape[0] != n_cols:
            raise ValueError(f"param length {arr.shape[0]} != n_cols {n_cols}")
        return arr

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


if __name__ == "__main__":
    adstocked = np.array([[0], [50000], [150000], [400000], [1000000]], dtype=float)
    sat = HillSaturation(k=150000, s=1.6)
    print(np.round(sat.fit_transform(adstocked), 3))
