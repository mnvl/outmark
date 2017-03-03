#! /usr/bin/python3

import logging
import numpy as np
from scipy import misc
from model5 import Model5
from datasets import CachingDataSet, CardiacDataSet
from preprocess import FeatureExtractor

def colorize(y):
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
  image = palette[y.astype(np.uint8).reshape(-1)]
  image = image.reshape((y.shape[0], y.shape[1], -1))
  return image

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='/dev/stderr',
                    filemode='w')


ds = CachingDataSet(CardiacDataSet())
fe = FeatureExtractor(ds, 5, 0)

settings = Model5.Settings()
settings.batch_size = 40
settings.num_classes = len(ds.get_classnames())
settings.D = 8
settings.W = 128
settings.H = 128
settings.kernel_size = 5
settings.num_conv_layers = 3
settings.num_conv_channels = 40
settings.num_dense_layers = 3
settings.num_dense_channels = 50
settings.learning_rate = 1e-4
model = Model5(settings)

validation_set_size = settings.batch_size
training_set_size = ds.get_size() - validation_set_size
val_accuracy = 0.
for i in range(50000):
  (X, y) = fe.get_examples(np.random.randint(0, training_set_size - 1, settings.batch_size),
                           settings.D, settings.H, settings.W)
  X = X.reshape(settings.batch_size, settings.D, settings.H, settings.W, 1)
  (loss, train_accuracy) = model.fit(X, y)

  if i % 10 == 0:
    (X_val, y_val) = fe.get_examples(np.random.randint(training_set_size, ds.get_size(), settings.batch_size),
                                     settings.D, settings.H, settings.W)
    X_val = X_val.reshape(settings.batch_size, settings.D, settings.H, settings.W, 1)
    predictions = model.predict(X_val)[0]
    pred_hard = (predictions > np.mean(predictions))

    image = X_val[0, settings.D // 2, :, :, 0]
    misc.imsave("debug/%06d_image.png" % i, image)
    eq_mask = (pred_hard[0, settings.D // 2, :, :].astype(np.uint8) == y_val[0, settings.D // 2, :, :])
    misc.imsave("debug/%06d_eq.png" % i, eq_mask)
    pred_mask = pred_hard[0, settings.D // 2, :, :]
    misc.imsave("debug/%06d_pred.png" % i, pred_mask)
    label_mask = y_val[0, settings.D // 2, :, :]
    misc.imsave("debug/%06d_label.png" % i, label_mask)
    mask = colorize(eq_mask * 1 + pred_mask * 2 + label_mask * 4)
    misc.imsave("debug/%06d_mask.png" % i, mask)
    misc.imsave("debug/%06d_mix.png" % i, np.expand_dims(image, 2) * mask)

    val_accuracy = val_accuracy * .5 + np.mean(pred_hard == y_val) * .5

    logging.info("step %d: accuracy = %f, loss = %f, val_accuracy = %f" % (i, train_accuracy, loss, val_accuracy))
  else:
    logging.info("step %d: accuracy = %f, loss = %f" % (i, train_accuracy, loss))
