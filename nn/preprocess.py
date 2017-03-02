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

    # self.image_unique_labels = {}
    # self.image_label_ijks = {}

  def get_random_example(self, D, H, W, image_index):
    image = self.dataset.get_training_set_image(image_index)
    label = self.dataset.get_training_set_label(image_index)

    # if image_index in self.image_unique_labels:
    #   unique_labels = self.image_unique_labels[image_index]
    # else:
    #   unique_labels = np.unique(label)
    #   self.image_unique_labels[image_index] = unique_labels

    # while True:
    #   selected_label = np.random.choice(unique_labels)

    #   if (image_index, selected_label) in self.image_label_ijks:
    #     ijks = self.image_label_ijks[(image_index, selected_label)]
    #   else:
    #     ijks = np.where((label == selected_label)[0:image.shape[0]-D, 0:image.shape[1]-H, 0:image.shape[2]-W])
    #     self.image_label_ijks[(image_index, selected_label)] = ijks

    #   if len(ijks[0]) == 0:
    #     continue

    #   selected_ijk = np.random.randint(0, len(ijks[0]))
    #   break

    # (i, j, k) = (ijks[0][selected_ijk], ijks[1][selected_ijk], ijks[2][selected_ijk])

    i = np.random.randint(0, image.shape[0] - D)
    j = np.random.randint(0, image.shape[1] - H)
    k = np.random.randint(0, image.shape[2] - W)

    logging.debug("Cube %s in %s." % (str((i, j, k)), str(image.shape)))

    X = image[i:i+D, j:j+H, k:k+W]
    y = label[i:i+D, j:j+H, k:k+W]

    return (X, y)

  # D, H, W should be odd.
  def get_random_training_example(self, D, H, W):
    image_index = np.random.randint(0, self.training_set_images)
    return self.get_random_example(D, H, W, image_index)

  # D, H, W should be odd.
  def get_random_training_batch(self, N, D, H, W):
    X = np.zeros(shape = (N, D, H, W))
    y = np.zeros(shape = (N, D, H, W))
    for i in range(N):
      (X[i], y[i]) = self.get_random_training_example(D, H, W)
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
