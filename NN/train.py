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
H = 256
W = 256
minibatch_size = 2
num_classes = len(ds.get_label_class_names())

cnn = CNN2({"D": D, "H":H, "W":W, "num_classes":num_classes, "minibatch_size":minibatch_size})

steps = 1000
val_accuracy_history = np.zeros(shape = steps)
for i in range(1000):
  (X, y) = fe.get_random_training_batch(minibatch_size, D, H, W)
  X = X.reshape(minibatch_size, D, H, W, 1)
  accuracy = cnn.fit(X, y)

  (X_val, y_val) = fe.get_random_validation_batch(minibatch_size, D, H, W)
  X_val = X_val.reshape(minibatch_size, D, H, W, 1)
  val_accuracy = cnn.evaluate(X_val, y_val)
  val_accuracy_history[i] = val_accuracy

  mean_val_accuracy = np.mean(val_accuracy_history[i-20:i])

  logging.info("step %d: accuracy = %f, val_accuracy = %f, mean_val_accuracy = %f" % (i, accuracy, val_accuracy, mean_val_accuracy))
