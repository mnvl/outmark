#! /usr/bin/python3

import random
import numpy as np
import logging
import unittest
import gflags
from scipy import misc

from datasets import RandomDataSet, CervixDataSet


class FeatureExtractor:

    def __init__(self, dataset):
        self.dataset = dataset
        self.N = self.dataset.get_size()
        self.C = len(dataset.get_classnames())

    def preprocess(self, image, label, D, H, W):
        label = label.astype(np.uint8)

        (d, h, w) = image.shape

        assert d >= D

        if d != D:
            i = random.randint(0, d - D)
            image = image[i: i + D, :, :]
            label = label[i: i + D, :, :]

        assert H == W
        if h > w:
            j = (h - w) // 2
            image = image[:, j: j + w, :]
            label = label[:, j: j + w, :]
        else:
            k = (w - h) // 2
            image = image[:, :, k: k + h]
            label = label[:, :, k: k + h]

        X = np.zeros((D, H, W))
        y = np.zeros((D, H, W), dtype=np.uint8)
        for i in range(D):
            X[i, :, :] = misc.imresize(image[i, :, :], (H, W), "bilinear")
            y[i, :, :] = misc.imresize(label[i, :, :], (H, W), "nearest")

        return (X, y)

    def get_example(self, index, D, H, W):
        (image, label) = self.dataset.get_image_and_label(index)
        assert image.shape == label.shape

        return self.preprocess(image, label, D, H, W)

    # D, H, W should be odd.
    def get_examples(self, image_indices, D, H, W):
        N = image_indices.shape[0]

        X = np.zeros(shape=(N, D, H, W))
        y = np.zeros(shape=(N, D, H, W))

        for i, index in enumerate(image_indices):
            (X[i, :, :, :], y[i, :, :, :]) = self.get_example(index, D, H, W)

        return X, y

    def get_images(self, image_indices, H, W):
        X = []
        y = []

        logging.info("loading images: " + str(image_indices))

        for index in image_indices:
            (image, label) = self.dataset.get_image_and_label(index)
            assert image.shape == label.shape

            (image, label) = self.preprocess(
                image, label, image.shape[0], H, W)

            X.append(image)
            y.append(label)

        assert len(X) > 0
        assert len(y) > 0

        return (X, y)


class TestFeatureExtractor(unittest.TestCase):

    def test_basic(self):
        fe = FeatureExtractor(RandomDataSet(10), 2, 2)
        (X, y) = fe.get_example(0, 9, 9, 9)
        fe.get_examples(np.array([0, 1, 2]), 9, 9, 9)

    def test_cervix(self):
        ds = CervixDataSet()
        fe = FeatureExtractor(ds, 10, 10)
        (X, y) = fe.get_example(5, 8, 128, 128)
        assert (y <= len(ds.get_classnames())).all(), str(np.unique(y))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename='/dev/stderr',
                        filemode='w')
    unittest.main()
