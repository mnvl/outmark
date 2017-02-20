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

class AbdomenDataset(Dataset):
  def __init__(self):
    self.training_set = []

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
    image_file = self.training_set[index][0]
    logging.info("reading image %s for Abdomen dataset." % image_file)
    image = nibabel.load(image_file)
    image_data = image.get_data()
    return image_data

  def get_training_set_label(self, index):
    label_file = self.training_set[index][1]
    logging.info("reading label %s for Abdomen dataset." % label_file)
    label = nibabel.load(label_file)
    label_data = label.get_data()
    return label_data

class TestAbdomenDataset(unittest.TestCase):
  def test_loading_training_set(self):
    abdomen = AbdomenDataset()
    for index in range(abdomen.get_training_set_size()):
      image = abdomen.get_training_set_image(index)
      label = abdomen.get_training_set_label(index)
      logging.info("Image shape is %s." % str(image.shape))
      assert image.shape == label.shape, image.shape + " != " + label.shape

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')
  unittest.main()
