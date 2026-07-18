"""
adstock.py
==========
Geometric adstock as an sklearn-compatible transformer.

    adstocked_t = spend_t + lambda * adstocked_{t-1}

lambda in [0, 1): higher = longer media memory (TV/influencer-style brand
carryover), lower = near-immediate decay (search-style intent capture).

Implemented as a TransformerMixin so it can sit inside a Pipeline/
ColumnTransformer and have its lambda tuned by the SAME hyperparameter
search that tunes saturation and the ElasticNet penalty — this is what lets
the whole pipeline fit "in one pass" instead of hand-tuning adstock per
channel before regression.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GeometricAdstock(BaseEstimator, TransformerMixin):
    """
    Applies per-column geometric adstock decay.

    Parameters
    ----------
    lambdas : dict[str, float] or float
        Decay rate per column (if dict, keyed by column name/index position)
        or a single float applied to all columns.
    """

    def __init__(self, lambdas=0.3):
        self.lambdas = lambdas

    def fit(self, X, y=None):
        # Stateless transform (decay rate is a hyperparameter, not learned
        # from data) -- fit() just validates shape for sklearn compatibility.
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n_rows, n_cols = X.shape

        if isinstance(self.lambdas, dict):
            lam_array = np.array([self.lambdas.get(i, 0.3) for i in range(n_cols)])
        elif np.isscalar(self.lambdas):
            lam_array = np.full(n_cols, self.lambdas)
        else:
            lam_array = np.asarray(self.lambdas, dtype=float)
            if lam_array.shape[0] != n_cols:
                raise ValueError(
                    f"lambdas length {lam_array.shape[0]} != n_cols {n_cols}"
                )

        out = np.zeros_like(X)
        carry = np.zeros(n_cols)
        for t in range(n_rows):
            carry = X[t, :] + lam_array * carry
            out[t, :] = carry
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


if __name__ == "__main__":
    # quick sanity check
    spend = np.array([[100, 0], [0, 50], [0, 0], [200, 0]], dtype=float)
    ad = GeometricAdstock(lambdas={0: 0.5, 1: 0.2})
    print(ad.fit_transform(spend))
