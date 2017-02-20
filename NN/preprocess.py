#! /usr/bin/python3

import numpy as np
import logging
import unittest
import gflags

from datasets import AbdomenDataset

class DatasetPreprocessor:
  def __init__(self, dataset):
    self.dataset = dataset

  # D, H, W should be odd.
  def get_random_example(self, D, H, W):
    image_index = np.random.randint(0, self.dataset.get_training_set_size())
    image = self.dataset.get_training_set_image(image_index)
    label = self.dataset.get_training_set_label(image_index)

    i = np.random.randint(0, image.shape[0] - D)
    j = np.random.randint(0, image.shape[1] - H)
    k = np.random.randint(0, image.shape[2] - W)

    X = image[i:i+D, j:j+H, k:k+W]
    y = label[i, j, k]

    return (X, y)

  def get_random_training_batch(self, N, D, H, W):
    X = np.zeros(shape = (N, D, H, W))
    y = np.zeros(shape = N)
    for i in range(N):
      (X[i], y[i]) = self.get_random_example(D, H, W)
    return (X, y)

class TestDataPreprocessor(unittest.TestCase):
  def test_abdomen(self):
    dp = DatasetPreprocessor(AbdomenDataset())
    (X, y) = dp.get_random_training_batch(100, 9, 9, 9)
    for i in range(100):
      print(np.mean(X[i]), y[i])

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')
  unittest.main()
