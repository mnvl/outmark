#! /usr/bin/python3

import random
import numpy as np
import logging
import unittest
import gflags

from datasets import RandomDataSet

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
    assert image.shape == label.shape, image.shape + " != " + label.shape

    i = random.randint(0, image.shape[0] - D)
    j = random.randint(0, image.shape[1] - H)
    k = random.randint(0, image.shape[2] - W)

    X = image[i:i+D, j:j+H, k:k+W]
    y = label[i:i+D, j:j+H, k:k+W]

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
  def test_random_batch(self):
    fe = FeatureExtractor(RandomDataset(10), 2, 2)
    (X, y) = fe.get_random_training_batch(100, 9, 9, 9)

  def test_random_batch(self):
    fe = FeatureExtractor(RandomDataset(10), 2, 2)
    (X, y) = fe.get_random_validation_batch(100, 9, 9, 9)

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')
  unittest.main()
