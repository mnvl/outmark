#! /usr/bin/python3

import random
import numpy as np
import logging
import unittest
import gflags
from scipy import misc

from datasets import RandomDataSet, CervixDataSet

class FeatureExtractor:
  def __init__(self, dataset, validation_set_images, test_set_images):
    self.dataset = dataset
    self.N = self.dataset.get_size()
    self.C = len(dataset.get_classnames())

    self.validation_set_images = validation_set_images
    self.test_set_images = test_set_images
    self.training_set_images = self.N - validation_set_images - test_set_images

  def get_example(self, index, D, H, W):
    (image, label) = self.dataset.get_image_and_label(index)
    assert image.shape == label.shape

    (d, h, w) = image.shape
    assert d >= D

    if d != D:
      i = random.randint(0, d - D)
      image = image[i : i + D, :, :]
      label = label[i : i + D, :, :]

    assert H == W
    if h > w:
      j = random.randint(0, h - w)
      image = image[:, j : j + w, :]
      label = label[:, j : j + w, :]
    else:
      k = random.randint(0, w - h)
      image = image[:, :, k : k + h]
      label = label[:, :, k : k + h]

    label = label.astype(np.uint8)

    X = np.zeros((D, H, W))
    y = np.zeros((D, H, W), dtype = np.uint8)
    for i in range(D):
      X[i, :, :] = misc.imresize(image[i, :, :], (H, W), "bilinear")
      y[i, :, :] = misc.imresize(label[i, :, :], (H, W), "nearest")

    return (X, y)

  # D, H, W should be odd.
  def get_examples(self, image_indices, D, H, W):
    N = image_indices.shape[0]

    X = np.zeros(shape = (N, D, H, W))
    y = np.zeros(shape = (N, D, H, W))

    for i, index in enumerate(image_indices):
      (X[i], y[i]) = self.get_example(index, D, H, W)

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
