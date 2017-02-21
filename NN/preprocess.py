#! /usr/bin/python3

import numpy as np
import logging
import unittest
import gflags

from datasets import RandomDataset

class FeatureExtractor:
  def __init__(self, dataset, validation_set_images, test_set_images):
    self.dataset = dataset
    self.N = self.dataset.get_training_set_size()
    self.C = len(dataset.get_label_class_names())

    self.validation_set_images = validation_set_images
    self.test_set_images = test_set_images
    self.training_set_images = self.N - validation_set_images - test_set_images

    self.image_unique_labels = {}
    self.image_label_ijks = {}

  # D, H, W should be odd.
  def get_random_example(self, D, H, W):
    image_index = np.random.randint(0, self.training_set_images)
    image = self.dataset.get_training_set_image(image_index)
    label = self.dataset.get_training_set_label(image_index)

    if image_index in self.image_unique_labels:
      unique_labels = self.image_unique_labels[image_index]
    else:
      unique_labels = np.unique(label)
      self.image_unique_labels[image_index] = unique_labels

    selected_label = np.random.choice(unique_labels)

    if (image_index, selected_label) in self.image_label_ijks:
      ijks = self.image_label_ijks[(image_index, selected_label)]
    else:
      ijks = np.where((label == selected_label)[0:image.shape[0]-D, 0:image.shape[1]-H, 0:image.shape[2]-W])
      self.image_label_ijks[(image_index, selected_label)] = ijks

    selected_ijk = np.random.randint(0, len(ijks[0]))

    (i, j, k) = (ijks[0][selected_ijk], ijks[1][selected_ijk], ijks[2][selected_ijk])

    X = image[i:i+D, j:j+H, k:k+W]

    y = np.zeros(shape = self.C)
    y[label[i, j, k]] = 1

    return (X, y)

  # D, H, W should be odd.
  def get_random_batch(self, N, D, H, W):
    X = np.zeros(shape = (N, D, H, W))
    y = np.zeros(shape = (N, self.C))
    for i in range(N):
      (X[i], y[i]) = self.get_random_example(D, H, W)
    return (X, y)

  # D, H, W should be odd.
  def get_validation_set(self, D, H, W):
    Xs = []
    ys = []

    for i in range(self.validation_set_images):
      index = i + self.training_set_images
      image = self.dataset.get_training_set_image(index)
      label = self.dataset.get_training_set_label(index)

      for i in range(0, image.shape[0] - D):
        for j in range(0, image.shape[1] - H):
          for k in range(0, image.shape[2] - W):
            X = image[i:i+D, j:j+H, k:k+W]

            y = np.zeros(shape = self.C)
            y[label[i, j, k]] = 1

            Xs.append(X)
            ys.append(y)

    Xs = np.stack(Xs, axis = -1)
    ys = np.stack(ys, axis = -1)

    return (Xs, ys)

class TestFeatureExtractor(unittest.TestCase):
  def test_random_batch(self):
    fe = FeatureExtractor(RandomDataset(10), 2, 2)
    (X, y) = fe.get_random_batch(100, 9, 9, 9)

  def test_random_batch(self):
    fe = FeatureExtractor(RandomDataset(2), 2, 0)
    (X, y) = fe.get_validation_set(9, 9, 9)

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')
  unittest.main()
