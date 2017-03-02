#! /usr/bin/python3

import logging
import numpy as np
from model5 import Model5
from datasets import AbdomenDataset
from preprocess import FeatureExtractor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='/dev/stderr',
                    filemode='w')


ds = AbdomenDataset()
fe = FeatureExtractor(ds, 5, 0)

settings = Model5.Settings()
settings.batch_size = 10

model = Model5(settings)

for i in range(1000):
  (X, y) = fe.get_random_training_batch(settings.batch_size, settings.D, settings.H, settings.W)
  X = X.reshape(settings.batch_size, settings.D, settings.H, settings.W, 1)
  (loss, train_accuracy) = model.fit(X, y)

  # TODO
  val_accuracy = 0

  logging.info("step %d: accuracy = %f, loss = %f, val_sample_accuracy = %f" % (i, train_accuracy, loss, val_accuracy))
