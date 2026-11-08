"""Classification metrics (Task 4). Class 1 (forged) is treated as 'positive'."""

import numpy as np


def confusion_counts(y_true, y_pred):
    """Return (tp, fp, tn, fn) counts."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def accuracy(y_true, y_pred):
    """Fraction of predictions that are correct. This will be the PSO fitness."""
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred):
    """Precision, recall and F1 for the positive class."""
    tp, fp, _, fn = confusion_counts(y_true, y_pred)
    # Guard against division by zero (e.g. network predicts all zeros).
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1
