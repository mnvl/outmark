#! /usr/bin/python3

import logging
import numpy as np
from model2 import CNN2
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

cnn = CNN2({
  "D": D, "H": H, "W": W, "num_classes": num_classes, "minibatch_size": minibatch_size,
  "conv1_size": 200, "conv2_size": 200, "conv3_size": 200, "conv4_size": 200,
  "l2_reg": 1e-4 })

steps = 1000
for i in range(1000):
  (X, y) = fe.get_random_training_batch(minibatch_size, D, H, W)
  X = X.reshape(minibatch_size, D, H, W, 1)
  (loss, train_accuracy) = cnn.fit(X, y)

  (X_val, y_val) = fe.get_random_validation_batch(minibatch_size, D, H, W)
  X_val = X_val.reshape(minibatch_size, D, H, W, 1)
  val_accuracy = cnn.evaluate(X_val, y_val)

  logging.info("step %d: accuracy = %f, loss = %f, val_sample_accuracy = %f" % (i, train_accuracy, loss, val_accuracy))
