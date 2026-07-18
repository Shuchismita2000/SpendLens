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

import numpy as np


def expanding_window_splits(n_samples: int, min_train_weeks: int, n_splits: int):
    """
    Yields (train_idx, test_idx) tuples. Train is always [0, train_end),
    test is the next contiguous block. Train grows each fold; test blocks
    are contiguous and non-overlapping across folds.
    """
    remaining = n_samples - min_train_weeks
    if remaining <= 0:
        raise ValueError(
            f"min_train_weeks ({min_train_weeks}) leaves no data to validate on "
            f"(n_samples={n_samples})."
        )
    step = max(1, remaining // n_splits)

    splits = []
    train_end = min_train_weeks
    for i in range(n_splits):
        test_start = train_end
        test_end = n_samples if i == n_splits - 1 else min(train_end + step, n_samples)
        if test_end <= test_start:
            break
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
        train_end = test_end
        if train_end >= n_samples:
            break
    return splits


if __name__ == "__main__":
    for tr, te in expanding_window_splits(130, min_train_weeks=52, n_splits=5):
        print(f"train=[0:{tr[-1]+1}] ({len(tr)} wks)  test=[{te[0]}:{te[-1]+1}] ({len(te)} wks)")
