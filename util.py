
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

def write_image_and_label(filename, image, label):
    with open(filename, "wb") as f:
        np.savez(f, image = image, label = label)

def read_image_and_label(filename, image, label):
    with open(filename, "rb") as f:
        data = np.loadz(f)
        return data["image"], data["label"]
