#! /usr/bin/python3

import logging
import numpy as np
from scipy import misc
from dense_unet import DenseUNet
from datasets import CachingDataSet, CardiacDataSet, CervixDataSet, AbdomenDataSet
from preprocess import FeatureExtractor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='train.log',
                    filemode='w')

ds = CardiacDataSet()
#ds = CervixDataSet()
#ds = AbdomenDataSet()
ds = CachingDataSet(ds)

fe = FeatureExtractor(ds, 5, 0)

settings = DenseUNet.Settings()
settings.batch_size = 1
settings.num_classes = len(ds.get_classnames())
settings.image_depth = 4
settings.image_width = 256
settings.image_height = 256
settings.learning_rate = 1e-5
model = DenseUNet(settings)
model.add_layers()
model.add_softmax_loss()
model.start()

validation_set_size = settings.batch_size
training_set_size = ds.get_size() - validation_set_size
val_accuracy = 0.
for i in range(50000):
  (X, y) = fe.get_examples(np.random.randint(0, training_set_size - 1, settings.batch_size),
                           settings.image_depth, settings.image_height, settings.image_width)

  # X = X[:, :, 128:384, 128:384]
  # y = y[:, :, 128:384, 128:384]
  X = X.reshape(settings.batch_size, settings.image_depth, settings.image_height, settings.image_width, 1)

  (loss, train_accuracy) = model.fit(X, y)

  if i % 10 == 0:
    (X_val, y_val) = fe.get_examples(np.random.randint(training_set_size, ds.get_size(), settings.batch_size),
                                     settings.image_depth, settings.image_height, settings.image_width)
    # X_val = X_val[:, :, 128:384, 128:384]
    # y_val = y_val[:, :, 128:384, 128:384]
    X_val = X_val.reshape(settings.batch_size, settings.image_depth, settings.image_height, settings.image_width, 1)
    predictions = model.predict(X_val)[0]

    image = X_val[0, settings.image_depth // 2, :, :, 0]
    misc.imsave("debug/%06d_0_image.png" % i, image)
    eq_mask = (predictions[0, settings.image_depth // 2, :, :].astype(np.uint8) == y_val[0, settings.image_depth // 2, :, :].astype(np.uint8))
    misc.imsave("debug/%06d_1_eq.png" % i, eq_mask)
    pred_mask = predictions[0, settings.image_depth // 2, :, :].astype(np.float32)
    misc.imsave("debug/%06d_2_pred.png" % i, pred_mask)
    label_mask = y_val[0, settings.image_depth // 2, :, :]
    misc.imsave("debug/%06d_3_label.png" % i, label_mask)
    mask = np.dstack((eq_mask, pred_mask, label_mask))
    misc.imsave("debug/%06d_4_mask.png" % i, mask)
    misc.imsave("debug/%06d_5_mix.png" % i, (100. + np.expand_dims(image, 2)) * (1. + mask))

    val_accuracy = val_accuracy * .5 + np.mean(predictions == y_val) * .5
    logging.info("step %d: accuracy = %f, loss = %f, val_accuracy = %f" % (i, train_accuracy, loss, val_accuracy))
    logging.info("predicted_labels = %s, actual_labels = %s" % (str(np.unique(predictions)), str(np.unique(y_val))))
  else:
    logging.info("step %d: accuracy = %f, loss = %f" % (i, train_accuracy, loss))
