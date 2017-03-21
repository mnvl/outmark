#! /usr/bin/python3

import sys
import logging
import numpy as np
from scipy import misc
import gflags
from unet import UNet
from dense_unet import DenseUNet
from datasets import CachingDataSet, CardiacDataSet, CervixDataSet, AbdomenDataSet
from preprocess import FeatureExtractor

gflags.DEFINE_boolean("notebook", False, "");
gflags.DEFINE_string("dataset", "Cardiac", "");

FLAGS = gflags.FLAGS

class Trainer:
  def __init__(self, settings, dataset, training_set_size, feature_extractor):
    self.dataset = dataset
    self.training_set_size = training_set_size
    self.feature_extractor = feature_extractor

    self.S = settings
    self.model = UNet(settings)
    self.model.add_layers()
    self.model.add_softmax_loss()

    self.train_loss_history = []
    self.train_accuracy_history = []
    self.val_accuracy_history = []
    self.val_dice_history = []

    self.model.start()

  def dice(self, a, b):
    assert a.shape == b.shape
    nominator = np.sum(((a != 0) * (b != 0) * (a == b)).astype(np.float32))
    denominator = float(np.sum(a != 0)) + np.sum(b != 0) + 1
    return nominator / denominator

  def train(self, num_steps, validate_every_steps = 200):
    # these are just lists of images as they can have mismatching depth dimensions
    logging.info("loading validation set")
    (val_images, val_labels) = fe.get_images(np.arange(self.training_set_size, self.dataset.get_size()),
                                   self.S.image_height, self.S.image_width)

    for step in range(num_steps):
      (X, y) = fe.get_examples(np.random.randint(0, self.training_set_size - 1, self.S.batch_size),
                               self.S.image_depth, self.S.image_height, self.S.image_width)
      X = np.expand_dims(X, axis = 4)

      (loss, train_accuracy, train_dice) = self.model.fit(X, y)

      self.train_loss_history.append(loss)
      self.train_accuracy_history.append(train_accuracy)

      if (step + 1) % validate_every_steps == 0:
        val_accuracy = []
        val_dice = []
        for i, (X_val, y_val) in enumerate(zip(val_images, val_labels)):
          X_val = np.expand_dims(X_val, axis = 4)
          y_pred = self.model.segment_image(X_val)
          val_accuracy.append(np.mean((y_pred == y_val).astype(np.float32)))
          val_dice.append(self.dice(y_pred, y_val))
        val_accuracy = np.mean(val_accuracy)
        val_dice = np.mean(val_dice)
        logging.info("step %d: accuracy = %f, dice = %f, loss = %f, val_accuracy = %f, val_dice = %f" % \
                     (step, train_accuracy, train_dice, loss, val_accuracy, val_dice))
        self.val_accuracy_history.append(val_accuracy)
        self.val_dice_history.append(val_dice)
      else:
        logging.info("step %d: accuracy = %f, dice = %f, loss = %f" % (step, train_accuracy, train_dice, loss))

    def clear(self):
      self.model.stop()

if __name__ == '__main__':
  FLAGS(sys.argv)

  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')

  if FLAGS.dataset == "Cardiac":
    ds = CardiacDataSet()
    ds = CachingDataSet(ds)
  elif FLAGS.dataset == "Cervix":
    ds = CervixDataSet()
  elif FLAGS.dataset == "Abdomen":
    ds = AbdomenDataSet()
  else:
    print("Unknown dataset: %s" % FLAGS.dataset, file = sys.stderr)
    sys.exit(1)

  fe = FeatureExtractor(ds, 5, 0)

  settings = UNet.Settings()
  settings.batch_size = 5
  settings.num_classes = len(ds.get_classnames())
  settings.class_weights = [1] + [10] * (settings.num_classes - 1)
  settings.image_depth = 1
  settings.image_width = 64 if FLAGS.notebook else 256
  settings.image_height = 64 if FLAGS.notebook else 256
  settings.num_conv_channels = 50
  settings.num_conv_blocks = 3
  settings.num_dense_channels = 100
  settings.learning_rate = 1e-6
  settings.l2_reg = 1e-4

  trainer = Trainer(settings, ds, max(ds.get_size()-20, 4*ds.get_size()//5), fe)
  trainer.train(1000, validate_every_steps = 50)
  trainer.clear()

# TODO:
# validation_set_size = settings.batch_size
# training_set_size = ds.get_size() - validation_set_size
# val_accuracy = 0.
# for i in range(50000):
#   (X, y) = fe.get_examples(np.random.randint(0, training_set_size - 1, settings.batch_size),
#                            settings.image_depth, settings.image_height, settings.image_width)



#   # X = X[:, :, 128:384, 128:384]
#   # y = y[:, :, 128:384, 128:384]

#   if i % 10 == 0:
#     (X_val, y_val) = fe.get_examples(np.random.randint(training_set_size, ds.get_size(), settings.batch_size),
#                                      settings.image_depth, settings.image_height, settings.image_width)
#     # X_val = X_val[:, :, 128:384, 128:384]
#     # y_val = y_val[:, :, 128:384, 128:384]
#     X_val = X_val.reshape(settings.batch_size, settings.image_depth, settings.image_height, settings.image_width, 1)
#     predictions = model.predict(X_val)[0]

#     image = X_val[0, settings.image_depth // 2, :, :, 0]
#     misc.imsave("debug/%06d_0_image.png" % i, image)
#     eq_mask = (predictions[0, settings.image_depth // 2, :, :].astype(np.uint8) == y_val[0, settings.image_depth // 2, :, :].astype(np.uint8))
#     misc.imsave("debug/%06d_1_eq.png" % i, eq_mask)
#     pred_mask = predictions[0, settings.image_depth // 2, :, :].astype(np.float32)
#     misc.imsave("debug/%06d_2_pred.png" % i, pred_mask)
#     label_mask = y_val[0, settings.image_depth // 2, :, :]
#     misc.imsave("debug/%06d_3_label.png" % i, label_mask)
#     mask = np.dstack((eq_mask, pred_mask, label_mask))
#     misc.imsave("debug/%06d_4_mask.png" % i, mask)
#     misc.imsave("debug/%06d_5_mix.png" % i, (100. + np.expand_dims(image, 2)) * (1. + mask))

#     val_accuracy = val_accuracy * .5 + np.mean(predictions == y_val) * .5
#     logging.info("step %d: accuracy = %f, loss = %f, val_accuracy = %f" % (i, train_accuracy, loss, val_accuracy))
#     logging.info("predicted_labels = %s, actual_labels = %s" % (str(np.unique(predictions)), str(np.unique(y_val))))
#   else:
#     logging.info("step %d: accuracy = %f, loss = %f" % (i, train_accuracy, loss))
