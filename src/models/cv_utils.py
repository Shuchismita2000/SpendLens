"""
cv_utils.py
===========

Utilities for expanding-window (walk-forward) cross-validation on time
series data.

Unlike random cross-validation, this approach preserves the chronological
order of observations so that future data is never used to predict the
past, preventing data leakage.

This module implements a lightweight alternative to scikit-learn's
`TimeSeriesSplit`. With relatively short marketing time series (e.g.,
around 130 weekly observations), `TimeSeriesSplit`'s evenly sized folds do
not provide precise control over the initial training window. In
particular, this implementation guarantees that the first fold contains at
least `min_train_weeks` of historical data before any validation occurs.

Each split guarantees:
    - The training set is a strict prefix of the time series (no shuffling
      or future leakage).
    - The first training fold contains at least `min_train_weeks` of
      history.
    - The training window expands monotonically across folds (never
      shrinks or slides).
    - The test set is the immediately following block, producing
      non-overlapping validation periods.
"""

import numpy as np


# =============================================================================
# Expanding Window Cross-Validation
# =============================================================================

def expanding_window_splits(
    n_samples: int,
    min_train_weeks: int,
    n_splits: int,
):
    """
    Generate expanding-window train/test splits for time series data.

    The training set always starts from the first observation and grows
    with each fold, while the test set consists of the immediately
    following contiguous observations.

    Example
    -------
    Fold 1:
        Train: [0 ............ 51]
        Test : [52 ......... 67]

    Fold 2:
        Train: [0 ............ 67]
        Test : [68 ......... 83]

    Parameters
    ----------
    n_samples : int
        Total number of observations.

    min_train_weeks : int
        Initial size of the training dataset.

    n_splits : int
        Number of cross-validation folds.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) tuples.

    Raises
    ------
    ValueError
        If the initial training period leaves no observations
        available for validation.
    """
    # Compute observations remaining for validation
    remaining = n_samples - min_train_weeks

    if remaining <= 0:
        raise ValueError(
            f"min_train_weeks ({min_train_weeks}) leaves no data "
            f"to validate on (n_samples={n_samples})."
        )

    # Determine approximate size of each test block
    step = max(1, remaining // n_splits)

    splits = []
    train_end = min_train_weeks

    # Generate expanding-window folds
    for i in range(n_splits):
        test_start = train_end

        test_end = (
            n_samples
            if i == n_splits - 1
            else min(train_end + step, n_samples)
        )

        # Stop if no validation observations remain
        if test_end <= test_start:
            break

        splits.append(
            (
                np.arange(0, train_end),
                np.arange(test_start, test_end),
            )
        )

        # Expand the training window
        train_end = test_end

        # Stop once all observations have been used
        if train_end >= n_samples:
            break

    return splits


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Generate example cross-validation folds
    splits = expanding_window_splits(
        n_samples=130,
        min_train_weeks=52,
        n_splits=5,
    )

    # Display each fold
    for train_idx, test_idx in splits:
        print(
            f"train=[0:{train_idx[-1] + 1}] "
            f"({len(train_idx)} weeks)   "
            f"test=[{test_idx[0]}:{test_idx[-1] + 1}] "
            f"({len(test_idx)} weeks)"
        )