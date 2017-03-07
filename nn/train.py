#! /usr/bin/python3

import logging
import numpy as np
from scipy import misc
from znet import ZNet
from datasets import CachingDataSet, CardiacDataSet
from preprocess import FeatureExtractor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='train.log',
                    filemode='w')

ds = CachingDataSet(CardiacDataSet())
fe = FeatureExtractor(ds, 5, 0)

settings = ZNet.Settings()
settings.batch_size = 5
settings.num_classes = len(ds.get_classnames())
settings.D = 8
settings.W = 128
settings.H = 128
settings.kernel_size = 5
settings.num_conv_layers = 5
settings.num_conv_channels = 40
settings.num_dense_layers = 3
settings.num_dense_channels = 50
settings.learning_rate = 1e-4
model = ZNet(settings)
model.add_layers()
model.add_softmax_loss()
model.init()

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
    misc.imsave("debug/%06d_0_image.png" % i, image)
    eq_mask = (pred_hard[0, settings.D // 2, :, :].astype(np.uint8) == y_val[0, settings.D // 2, :, :])
    misc.imsave("debug/%06d_1_eq.png" % i, eq_mask)
    pred_mask = pred_hard[0, settings.D // 2, :, :]
    misc.imsave("debug/%06d_2_pred.png" % i, pred_mask)
    label_mask = y_val[0, settings.D // 2, :, :]
    misc.imsave("debug/%06d_3_label.png" % i, label_mask)
    mask = np.dstack((eq_mask, pred_mask, label_mask))
    misc.imsave("debug/%06d_4_mask.png" % i, mask)
    misc.imsave("debug/%06d_5_mix.png" % i, (100. + np.expand_dims(image, 2)) * (1. + mask))

    val_accuracy = val_accuracy * .5 + np.mean(pred_hard == y_val) * .5

    logging.info("step %d: accuracy = %f, loss = %f, val_accuracy = %f" % (i, train_accuracy, loss, val_accuracy))
  else:
    logging.info("step %d: accuracy = %f, loss = %f" % (i, train_accuracy, loss))
