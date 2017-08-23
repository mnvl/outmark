
import numpy as np
import logging


class AttributeDict(dict):

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

        def __delattr__(self, key):
            del self[key]


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


def text_hist(a, bins=40):
    h, b = np.histogram(a, bins)
    scale = 80.0 / np.amax(h)
    text = ""
    for i in range(0, bins - 1):
        text += ("%8.2f | %10d | %s\n" %
                 (b[i], h[i - 1], '*' * int(scale * h[i - 1])))
    text += "%8.2f" % b[bins]
    return text


# window_size should be odd
def moving_average(a, window_size=9):
    a = np.concatenate(
        (np.repeat(a[0], window_size // 2), a, np.repeat(a[-1], window_size // 2)))
    W = np.ones(window_size) / float(window_size)
    return np.convolve(a, W, "valid")


def setup_logging():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename='/dev/stderr',
                        filemode='w')
