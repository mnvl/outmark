#! /usr/bin/python3

# WARNING: this code is super slow, look for a C++ version of same code.

import os
import sys
from glob import glob
import gflags
import numpy as np
import nibabel

FLAGS = gflags.FLAGS
FLAGS(sys.argv)

gflags.DEFINE_string("image_dir", "/large/data/Abdomen/RawData/Training/img", "");
gflags.DEFINE_string("image_prefix", "img", "");
gflags.DEFINE_string("label_dir", "/large/data/Abdomen/RawData/Training/label", "");
gflags.DEFINE_string("label_prefix", "label", "");

deltas = [
  [ 0, 0, 0],
  [-1, 0, 0],
  [ 1, 0, 0],
  [ 0,-1, 0],
  [ 0, 1, 0],
  [ 0, 0,-1],
  [ 0, 0, 1]]

for image_file in glob(os.path.join(FLAGS.image_dir,  FLAGS.image_prefix + "*.nii")):
  label_file = os.path.join(FLAGS.label_dir, os.path.basename(image_file).replace(FLAGS.image_prefix, FLAGS.label_prefix))
  print("\n***", image_file, label_file)

  image = nibabel.load(image_file)
  label = nibabel.load(label_file)
  print("image", image.shape, image.get_data_dtype())
  print("label", label.shape, label.get_data_dtype())

  assert (image.shape == label.shape)

  image_data = image.get_data()
  label_data = label.get_data()

  for i in range(1, image.shape[0] - 1):
    print("%.2f%% done" % (i / (image.shape[0] - 2) * 100))
    for j in range(1, image.shape[1] - 1):
      for k in range(1, image.shape[2] - 1):
        pos = np.array([i, j, k])
        features = [label_data[tuple(pos)]]
        
        for delta in deltas:
          features.append((image_data[tuple(pos + delta)]))
