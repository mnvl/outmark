#! /usr/bin/python3

import logging
import numpy as np
from model2 import CNN2
from model3 import CNN3
from datasets import AbdomenDataset
from preprocess import FeatureExtractor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='/dev/stderr',
                    filemode='w')


ds = AbdomenDataset()
fe = FeatureExtractor(ds, 5, 0)

D = 16
H = 64
W = 64
minibatch_size = 10
num_classes = len(ds.get_label_class_names())

# cnn = CNN2({
#   "D": D, "H": H, "W": W, "num_classes": num_classes, "minibatch_size": minibatch_size })

cnn = CNN3({
  "D": D, "H": H, "W": W,
  "num_classes": num_classes,
  "minibatch_size": minibatch_size,
  "l2_reg": 1e-4, "dropout": 0.9 })

for i in range(1000):
  (X, y) = fe.get_random_training_batch(minibatch_size, D, H, W)
  X = X.reshape(minibatch_size, D, H, W, 1)
  (loss, train_accuracy) = cnn.fit(X, y)

  # TODO
  val_accuracy = None

  logging.info("step %d: accuracy = %f, loss = %f, val_sample_accuracy = %f" % (i, train_accuracy, loss, val_accuracy))
