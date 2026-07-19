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

"""
Hill Saturation Transformer

This module implements a custom scikit-learn transformer that applies
the Hill saturation function to marketing variables.

The Hill function models diminishing returns in marketing spend, where
additional investment produces progressively smaller incremental effects
after a certain spending threshold.

The transformer is fully compatible with scikit-learn pipelines.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# =============================================================================
# Hill Saturation Transformer
# =============================================================================

class HillSaturation(BaseEstimator, TransformerMixin):
    """
    Apply the Hill saturation transformation to one or more features.

    The Hill function is defined as:

        y = x^s / (x^s + k^s)

    where:
        - x is the input value.
        - k is the half-saturation point.
        - s controls the steepness of the curve.

    Parameters
    ----------
    k : float | list | dict, default=100000.0
        Half-saturation value(s).

        - float:
            Same value applied to every feature.

        - list or ndarray:
            One value per feature.

        - dict:
            Mapping of column index to half-saturation value.
            Unspecified columns default to 100000.0.

    s : float | list | dict, default=1.5
        Steepness parameter(s).

        - float:
            Same value applied to every feature.

        - list or ndarray:
            One value per feature.

        - dict:
            Mapping of column index to steepness value.
            Unspecified columns default to 1.5.
    """

    def __init__(self, k=100000.0, s=1.5):
        self.k = k
        self.s = s

    def fit(self, X, y=None):
        """
        Validate the input and store the number of features.

        This transformer is stateless—the saturation parameters are
        predefined hyperparameters rather than learned from the data.

        Parameters
        ----------
        X : array-like
            Input feature matrix.
        y : ignored
            Included for scikit-learn compatibility.

        Returns
        -------
        HillSaturation
            Fitted transformer.
        """
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        """
        Apply the Hill saturation transformation.

        Parameters
        ----------
        X : array-like
            Input feature matrix.

        Returns
        -------
        ndarray
            Saturated feature matrix.
        """
        X = np.asarray(X, dtype=float)
        n_rows, n_cols = X.shape

        # Marketing spend cannot be negative
        X = np.clip(X, 0, None)

        # Convert parameters into per-feature arrays
        k_arr = self._to_array(
            self.k,
            n_cols,
            default=100000.0,
        )

        s_arr = self._to_array(
            self.s,
            n_cols,
            default=1.5,
        )

        # Prevent division by zero
        k_arr = np.where(k_arr <= 0, 1.0, k_arr)

        # Apply Hill saturation equation
        with np.errstate(invalid="ignore", divide="ignore"):
            numer = X ** s_arr
            denom = numer + k_arr ** s_arr

            out = np.divide(
                numer,
                denom,
                out=np.zeros_like(X),
                where=denom > 0,
            )

        return out

    @staticmethod
    def _to_array(param, n_cols, default):
        """
        Convert a scalar, dictionary, or iterable parameter into
        a NumPy array with one value per feature.

        Parameters
        ----------
        param : float | list | dict
            Input parameter specification.
        n_cols : int
            Number of input features.
        default : float
            Default value for unspecified dictionary entries.

        Returns
        -------
        ndarray
            Parameter array.

        Raises
        ------
        ValueError
            If an iterable parameter has an incorrect length.
        """
        if isinstance(param, dict):
            return np.array(
                [
                    param.get(i, default)
                    for i in range(n_cols)
                ]
            )

        elif np.isscalar(param):
            return np.full(n_cols, param)

        arr = np.asarray(param, dtype=float)

        if arr.shape[0] != n_cols:
            raise ValueError(
                f"param length {arr.shape[0]} != n_cols {n_cols}"
            )

        return arr

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names.

        Since saturation preserves the number and order of features,
        the original feature names are returned unchanged.

        Parameters
        ----------
        input_features : array-like, optional
            Original feature names.

        Returns
        -------
        ndarray
            Output feature names.
        """
        return np.asarray(input_features)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example adstocked marketing spend
    adstocked = np.array(
        [
            [0],
            [50000],
            [150000],
            [400000],
            [1000000],
        ],
        dtype=float,
    )

    # Create Hill saturation transformer
    sat = HillSaturation(
        k=150000,
        s=1.6,
    )

    # Apply transformation
    transformed = sat.fit_transform(adstocked)

    print(np.round(transformed, 3))
