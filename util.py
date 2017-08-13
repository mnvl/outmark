
import numpy as np


def accuracy(a, b):
    return np.mean((a.flatten() == b.flatten()).astype(np.float32))


def iou(a, b, num_classes):
    assert a.shape == b.shape

    a = a.flatten()
    b = b.flatten()

    s = 0.0
    for c in range(1, num_classes):
        i = np.sum(np.logical_and(a == c, b == c), dtype=np.float32)
        u = np.sum(np.logical_or(a == c, b == c), dtype=np.float32)
        s += i / (u + 1.0)

    return s / (num_classes - 1.0)


def crappyhist(a, bins=40):
    h, b = np.histogram(a, bins)
    text = []

    for i in range(0, bins - 1):
        text.append("%8.2f | %10d | %s" %
                    (b[i], h[i - 1], '*' * int(70 * h[i - 1] / np.amax(h))))

    text.append("%8.2f" % (b[bins]))

    return "\n".join(text)
