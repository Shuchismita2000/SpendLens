"""
cv_utils.py
===========
Expanding-window (walk-forward) CV splitter.

Why not sklearn's TimeSeriesSplit directly: with only 130 weeks and a
requirement that the FIRST fold already has >= min_train_weeks of history
(no validating a model trained on 10 weeks of data), TimeSeriesSplit's
even-split logic doesn't give clean control over the first fold's size.
This is a thin, explicit alternative that guarantees:
    - every fold's train set is a strict PREFIX of the data (no shuffling,
      no future leakage)
    - the first fold has at least `min_train_weeks` of history
    - training window only grows across folds (never shrinks/slides)
"""

"""
Time Series Cross-Validation Utilities

This module provides an expanding window cross-validation strategy
for time series datasets.

Unlike random cross-validation, the expanding window approach preserves
the chronological order of observations, ensuring that future data is
never used to predict the past.

Each split:
    - Expands the training set.
    - Uses the immediately following block as the test set.
    - Produces non-overlapping test sets.
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