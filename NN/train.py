#! /usr/bin/python3

import logging
from model import CNN
from datasets import AbdomenDataset
from preprocess import FeatureExtractor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='/dev/stderr',
                    filemode='w')


ds = AbdomenDataset()
fe = FeatureExtractor(ds, 5, 0)

D = 9
H = 19
W = 19
minibatch_size = 500
num_classes = len(ds.get_label_class_names())

conv1_size = 100
conv2_size = 100
conv3_size = 100
fc1_size = 200
fc2_size = 200

cnn = CNN(D, H, W, minibatch_size, num_classes, conv1_size, conv2_size, conv3_size, fc1_size, fc2_size)

for i in range(1000):
  (X, y) = fe.get_random_training_batch(minibatch_size, D, H, W)
  (X_val, y_val) = fe.get_random_validation_batch(minibatch_size, D, H, W)

  X = X.reshape(minibatch_size, D, H, W, 1)
  X_val = X_val.reshape(minibatch_size, D, H, W, 1)
  (accuracy, val_accuracy) = cnn.fit(X, y, X_val, y_val)

  logging.info("step %d: accuracy = %f, val_accuracy = %f" % (i, accuracy, val_accuracy))
