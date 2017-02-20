#! /usr/bin/python3

import os
import sys
from glob import glob
import logging
import unittest
import gflags
import numpy as np
import nibabel

FLAGS = gflags.FLAGS
FLAGS(sys.argv)

gflags.DEFINE_string("abdomen_training_image_dir", "/large/data/Abdomen/RawData/Training/img", "");
gflags.DEFINE_string("abdomen_training_label_dir", "/large/data/Abdomen/RawData/Training/label", "");
gflags.DEFINE_string("abdomen_image_prefix", "img", "");
gflags.DEFINE_string("abdomen_label_prefix", "label", "");

class Dataset:
  def get_training_set_size(self):
    raise NotImplementedError

  def get_training_set_image(self, index):
    raise NotImplementedError

  def get_training_set_label(self, index):
    raise NotImplementedError

  def get_label_class_names(self):
    raise NotImplementedError

  def clear_cache(self):
    pass

class AbdomenDataset(Dataset):
  def __init__(self):
    self.training_set = []
    self.cache = {}

    for image_file in glob(os.path.join(FLAGS.abdomen_training_image_dir,
                                        FLAGS.abdomen_image_prefix + "*.nii.gz")):
      logging.info("Found image %s in Abdomen dataset." % image_file)
      label_file = os.path.join(FLAGS.abdomen_training_label_dir,
                                os.path.basename(image_file).replace(FLAGS.abdomen_image_prefix,
                                                                     FLAGS.abdomen_label_prefix))
      self.training_set.append((image_file, label_file))
    assert len(self.training_set) > 0, "No images found in Abdomen dataset."

  def get_training_set_size(self):
    return len(self.training_set)

  def get_training_set_image(self, index):
    (image_file, label_file) = self.training_set[index]
    if image_file in self.cache:
      logging.debug("Cache hit for image %s for Abdomen dataset." % image_file)
      image = self.cache[image_file]
    else:
      logging.info("Reading image %s for Abdomen dataset." % image_file)
      image = nibabel.load(image_file)
      self.cache[image_file] = image
    image_data = image.get_data()
    return image_data

  def get_training_set_label(self, index):
    (image_file, label_file) = self.training_set[index]
    if label_file in self.cache:
      logging.debug("Cache hit for label %s for Abdomen dataset." % label_file)
      label = self.cache[label_file]
    else:
      logging.info("Reading label %s for Abdomen dataset." % label_file)
      label = nibabel.load(label_file)
      self.cache[label_file] = label
    label_data = label.get_data()
    return label_data

  def get_label_class_names(self):
    return [
      "(0) none",
      "(1) spleen",
      "(2) right kidney",
      "(3) left kidney",
      "(4) gallbladder",
      "(5) esophagus",
      "(6) liver",
      "(7) stomach",
      "(8) aorta",
      "(9) inferior vena cava",
      "(10) portal vein and splenic vein",
      "(11) pancreas",
      "(12) right adrenal gland",
      "(13) left adrenal gland",
    ]

  def clear_cache(self):
    self.cache = None

class TestAbdomenDataset(unittest.TestCase):
  def test_loading_training_set(self):
    abdomen = AbdomenDataset()
    for index in range(abdomen.get_training_set_size()):
      image = abdomen.get_training_set_image(index)
      label = abdomen.get_training_set_label(index)
      logging.info("Image shape is %s." % str(image.shape))
      assert image.shape == label.shape, image.shape + " != " + label.shape

datasets = {
  "Abdomen": AbdomenDataset
}

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')
  unittest.main()
