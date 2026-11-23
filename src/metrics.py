import numpy as np


def confusion_counts(y_true, y_pred):
    # forged (1) is the positive class
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def accuracy(y_true, y_pred):
    # also used as the PSO fitness
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred):
    # precision = of the notes called forged, how many really were forged
    # recall = of the really forged notes, how many were found
    # f1 = balance of the two (harmonic mean)
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0  # nothing predicted as forged

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1
