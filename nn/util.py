
import numpy as np


def accuracy(a, b):
    return np.mean((a.flatten() == b.flatten()).astype(np.float32))


def iou(a, b):
    assert a.shape == b.shape

    a = a.flatten()
    b = b.flatten()

    a_nonzero = (a != 0).astype(np.float32)
    b_nonzero = (b != 0).astype(np.float32)

    intersection = (a == b).astype(np.float32)
    intersection = np.multiply(intersection, a_nonzero)
    intersection = np.multiply(intersection, b_nonzero)
    intersection = np.sum(intersection)

    union = np.sum(a_nonzero) + np.sum(b_nonzero) - intersection

    return (intersection + 1.0) / (union + 1.0)
