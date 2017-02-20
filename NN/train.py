#! /usr/bin/python3

import logging
from model import CNN
from datasets import AbdomenDataset
from preprocess import DatasetPreprocessor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='/dev/stderr',
                    filemode='w')


ds = AbdomenDataset()
pp = DatasetPreprocessor(ds)

D = 17
H = 9
W = 17
minibatch_size = 100
num_classes = len(ds.get_label_class_names())

conv1_size = 100
conv2_size = 100
fc1_size = 200
fc2_size = 200

cnn = CNN(D, H, W, minibatch_size, num_classes, conv1_size, conv2_size, fc1_size, fc2_size)

for i in range(1000):
  (X, y) = pp.get_random_training_batch(minibatch_size, D, H, W)
  X = X.reshape(minibatch_size, D, H, W, 1)
  accuracy = cnn.fit(X, y)
  if i % 10 == 0: logging.info("step %d: accuracy = %f" % (i, accuracy))
