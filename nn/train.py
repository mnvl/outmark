#! /usr/bin/python3

import logging
import numpy as np
from scipy import misc
from model5 import Model5
from datasets import AbdomenDataset
from preprocess import FeatureExtractor

def colorize(X, y):
  palette = np.array([
    [ 0x85, 0x99, 0x00 ],
    [ 0x26, 0x8b, 0xd2 ],
    [ 0x00, 0x2b, 0x36 ],
    [ 0x07, 0x36, 0x42 ],
    [ 0x58, 0x6e, 0x75 ],
    [ 0x65, 0x7b, 0x83 ],
    [ 0x83, 0x94, 0x96 ],
    [ 0x93, 0xa1, 0xa1 ],
    [ 0xee, 0xe8, 0xd5 ],
    [ 0xfd, 0xf6, 0xe3 ],
    [ 0xb5, 0x89, 0x00 ],
    [ 0xcb, 0x4b, 0x16 ],
    [ 0xdc, 0x32, 0x2f ],
    [ 0xd3, 0x36, 0x82 ],
    [ 0x6c, 0x71, 0xc4 ],
    [ 0x26, 0x8b, 0xd2 ],
    [ 0x2a, 0xa1, 0x98 ],
  ])

  image = np.expand_dims(X.reshape(-1), 1) * palette[y.astype(np.uint8).reshape(-1)]
  image = image.reshape((X.shape[0], X.shape[1], 3))
  return image

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='/dev/stderr',
                    filemode='w')


ds = AbdomenDataset()
fe = FeatureExtractor(ds, 5, 0)

settings = Model5.Settings()
settings.batch_size = 1

model = Model5(settings)

validation_set_size = settings.batch_size
training_set_size = ds.get_size() - validation_set_size
val_accuracy = 0.
for i in range(1000):
  (X, y) = fe.get_examples(np.random.randint(0, training_set_size - 1, settings.batch_size),
                           settings.D, settings.H, settings.W)
  X = X.reshape(settings.batch_size, settings.D, settings.H, settings.W, 1)
  (loss, train_accuracy) = model.fit(X, y)

  if i % 10 == 0:
    (X_val, y_val) = fe.get_examples(np.random.randint(training_set_size, ds.get_size(), settings.batch_size),
                                     settings.D, settings.H, settings.W)
    X_val = X_val.reshape(settings.batch_size, settings.D, settings.H, settings.W, 1)
    predictions = model.predict(X_val)

    misc.imsave("debug/%05d_val.png" % i, colorize(X_val[0, settings.D // 2, :, :, 0],
                                                   predictions[0][0, settings.D // 2, :, :]))

    misc.imsave("debug/%05d_pred.png" % i, colorize(X_val[0, settings.D // 2, :, :, 0],
                                                    y_val[0, settings.D // 2, :, :]))

    misc.imsave("debug/%05d_match.png" % i, colorize(X_val[0, settings.D // 2, :, :, 0],
                                                    (predictions[0][0, settings.D // 2, :, :] == y_val[0, settings.D // 2, :, :]).astype(np.uint8)))

    val_accuracy = val_accuracy * .5 + np.mean(predictions == y_val) * .5

    logging.info("step %d: accuracy = %f, loss = %f, val_accuracy = %f" % (i, train_accuracy, loss, val_accuracy))
  else:
    logging.info("step %d: accuracy = %f, loss = %f" % (i, train_accuracy, loss))
