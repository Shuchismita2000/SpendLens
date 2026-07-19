"""
adstock.py
==========

Geometric adstock transformer for Marketing Mix Modeling (MMM).

This module implements a custom scikit-learn-compatible transformer that
models the carryover effect of advertising, where marketing impact decays
gradually over time rather than disappearing immediately.

The transformation follows:

    adstocked_t = spend_t + λ * adstocked_{t-1}

where λ ∈ [0, 1). Larger values represent longer-lasting advertising
effects (e.g., TV or influencer campaigns), while smaller values represent
rapidly decaying channels such as paid search.

Implemented as a `TransformerMixin`, the transformer integrates directly
into scikit-learn `Pipeline` and `ColumnTransformer` workflows. This allows
the adstock decay parameter (λ) to be jointly optimized alongside Hill
saturation parameters and ElasticNet regularization during hyperparameter
search, enabling the entire MMM pipeline to be trained in a single,
end-to-end optimization rather than requiring manual adstock tuning before
regression.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# =============================================================================
# Geometric Adstock Transformer
# =============================================================================

class GeometricAdstock(BaseEstimator, TransformerMixin):
    """
    Apply geometric adstock transformation to one or more features.

    Each feature is transformed independently using the recursive formula:

        Adstock_t = X_t + λ × Adstock_(t−1)

    where:
        - X_t is the current value.
        - λ (lambda) is the decay rate.
        - Adstock_(t−1) is the previous accumulated effect.

    Parameters
    ----------
    lambdas : float | list | dict, default=0.3
        Decay rate(s) for the transformation.

        - float:
            Same decay applied to every feature.

        - list or ndarray:
            One decay value per feature.

        - dict:
            Mapping of column index to decay value.
            Unspecified columns default to 0.3.
    """

    def __init__(self, lambdas=0.3):
        self.lambdas = lambdas

    def fit(self, X, y=None):
        """
        Validate the input and store the number of features.

        This transformer is stateless—the decay rates are predefined
        hyperparameters rather than learned from the data.

        Parameters
        ----------
        X : array-like
            Input feature matrix.
        y : ignored
            Included for scikit-learn compatibility.

        Returns
        -------
        GeometricAdstock
            Fitted transformer.
        """
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        """
        Apply geometric adstock transformation.

        Parameters
        ----------
        X : array-like
            Input feature matrix.

        Returns
        -------
        ndarray
            Adstock-transformed feature matrix.

        Raises
        ------
        ValueError
            If the number of supplied decay rates does not match the
            number of input features.
        """
        X = np.asarray(X, dtype=float)
        n_rows, n_cols = X.shape

        # Determine decay rate for each feature
        if isinstance(self.lambdas, dict):
            lam_array = np.array(
                [self.lambdas.get(i, 0.3) for i in range(n_cols)]
            )

        elif np.isscalar(self.lambdas):
            lam_array = np.full(n_cols, self.lambdas)

        else:
            lam_array = np.asarray(self.lambdas, dtype=float)

            if lam_array.shape[0] != n_cols:
                raise ValueError(
                    f"lambdas length {lam_array.shape[0]} "
                    f"!= n_cols {n_cols}"
                )

        # Initialize output array and carryover values
        out = np.zeros_like(X)
        carry = np.zeros(n_cols)

        # Apply recursive adstock equation
        for t in range(n_rows):
            carry = X[t, :] + lam_array * carry
            out[t, :] = carry

        return out

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names.

        Since adstock does not change the number or order of features,
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
    # Example weekly marketing spend
    spend = np.array(
        [
            [100, 0],
            [0, 50],
            [0, 0],
            [200, 0],
        ],
        dtype=float,
    )

    # Create transformer with feature-specific decay rates
    ad = GeometricAdstock(
        lambdas={
            0: 0.5,
            1: 0.2,
        }
    )

    # Apply transformation
    transformed = ad.fit_transform(spend)

    print(transformed)