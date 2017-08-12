#! /usr/bin/python3

import numpy as np


def accuracy(a, b):
    return np.mean((a.flatten() == b.flatten()).astype(np.float32))


def iou(a, b, num_classes):
    assert a.shape == b.shape

    a = a.flatten()
    b = b.flatten()

    s = 0.0
    for c in range(1, num_classes):
        i = np.sum(np.logical_and(a == c, b == c), dtype = np.float32)
        u = np.sum(np.logical_or(a == c, b == c), dtype = np.float32)
        s += i / (u + 1.0)

    return s / (num_classes - 1.0)
