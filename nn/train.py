#! /usr/bin/python3

import sys
import logging
import random
import numpy as np
import tensorflow as tf
from scipy import misc
import gflags
from unet import UNet
from dense_unet import DenseUNet
from datasets import CachingDataSet, CardiacDataSet, CervixDataSet, AbdomenDataSet
from preprocess import FeatureExtractor
import util

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

    self.dataset_shuffle = np.arange(dataset.get_size())
    np.random.shuffle(self.dataset_shuffle)

    self.model.start()

  def train(self, num_steps, estimate_every_steps = 25, validate_every_steps = 200):
    val_accuracy_estimate = 0
    val_dice_estimate = 0

    # these are just lists of images as they can have mismatching depth dimensions

    logging.info("loading validation set")
    (val_images, val_labels) = fe.get_images(
      self.dataset_shuffle[np.arange(self.training_set_size, self.dataset.get_size())],
      self.S.image_height, self.S.image_width)
    
    for step in range(num_steps):
      (X, y) = fe.get_examples(
        self.dataset_shuffle[np.random.randint(0, self.training_set_size - 1, self.S.batch_size)],
        self.S.image_depth, self.S.image_height, self.S.image_width)
      X = np.expand_dims(X, axis = 4)

      (loss, train_accuracy, train_dice) = self.model.fit(X, y)

      self.train_loss_history.append(loss)
      self.train_accuracy_history.append(train_accuracy)

      if (step + 1) % validate_every_steps == 0 or step == 0:
        val_accuracy = []
        pred_labels = []
        for i, (X_val, y_val) in enumerate(zip(val_images, val_labels)):
          X_val = np.expand_dims(X_val, axis = 4)
          y_pred = self.model.segment_image(X_val)
          pred_labels.append(y_pred)

        pred_labels_flat = np.concatenate([x.flatten() for x in pred_labels])
        val_labels_flat = np.concatenate([x.flatten() for x in val_labels])

        val_accuracy = util.accuracy(pred_labels_flat, val_labels_flat)
        val_dice = util.dice(pred_labels_flat, val_labels_flat)

        logging.info("step %d: accuracy = %f, dice = %f, loss = %f, val_accuracy = %f, val_dice = %f" % \
                     (step, train_accuracy, train_dice, loss, val_accuracy, val_dice))

        self.val_accuracy_history.append(val_accuracy)
        self.val_dice_history.append(val_dice)
      elif (step + 1) % estimate_every_steps == 0:
        (X_val, y_val) = fe.get_examples(
          self.dataset_shuffle[np.random.randint(self.training_set_size, self.dataset.get_size() - 1, self.S.batch_size)],
          self.S.image_depth, self.S.image_height, self.S.image_width)
        X_val = np.expand_dims(X_val, axis = 4)

        y_pred = self.model.predict(X_val)

        val_accuracy_estimate = val_accuracy_estimate * 0.5 + util.accuracy(y_pred, y_val) * 0.5
        val_dice_estimate = val_dice_estimate * 0.5 + util.dice(y_pred, y_val) * 0.5

        logging.info("step %d: accuracy = %f, dice = %f, loss = %f, val_accuracy_estimate = %f, val_dice_estimate = %f" % \
                     (step, train_accuracy, train_dice, loss, val_accuracy_estimate, val_dice_estimate))
      else:
        logging.info("step %d: accuracy = %f, dice = %f, loss = %f" % (step, train_accuracy, train_dice, loss))

  def clear(self):
    self.model.stop()

def make_settings(fiddle = False):
  settings = UNet.Settings()
  settings.batch_size = 4
  settings.num_classes = len(ds.get_classnames())
  settings.class_weights = [1] + [random.uniform(1, 3) if fiddle else 10] * (settings.num_classes - 1)
  settings.image_depth = random.choice([1]) if fiddle else 1
  settings.image_width = 64 if FLAGS.notebook else 256
  settings.image_height = 64 if FLAGS.notebook else 256
  settings.kernel_size = random.choice([3, 5, 7]) if fiddle else 5
  settings.num_conv_channels = random.randint(20, 80) if fiddle else 10
  settings.num_conv_layers_per_block = random.randint(1, 3) if fiddle else 2
  settings.num_conv_blocks = random.randint(1, 3) if fiddle else 2
  settings.num_dense_channels = random.randint(50, 200) if fiddle else 100
  settings.num_dense_layers = random.randint(1, 5) if fiddle else 2
  settings.learning_rate = 0.0001 * (10**random.uniform(-2, 2) if fiddle else 1)
  settings.l2_reg = 100 * (10**random.uniform(-2, 2) if fiddle else 1.)
  settings.use_batch_norm = random.choice([True, False]) if fiddle else False
  settings.keep_prob = 0.1
  return settings

def search_for_best_settings(ds, fe):
  best_dice = -1
  best_dice_settings = None
  best_accuracy = -1
  best_accuracy_settings = None

  for i in range(100):
    settings = make_settings(fiddle = True)

    logging.info("try %d, settings: %s" % (i, str(vars(settings))))

    try:
      trainer = Trainer(settings, ds, 4*ds.get_size()//5, fe)
      trainer.train(1000)
    except tf.errors.ResourceExhaustedError as e:
      trainer.clear()
      logging.info("Resource exhausted: %s", e.message)
      continue
    finally:
      trainer.clear()

    logging.info("dice = %f, best_dice = %f" % (trainer.val_dice_history[-1], best_dice))
    if best_dice < trainer.val_dice_history[-1]:
      best_dice = trainer.val_dice_history[-1]
      best_dice_settings = settings
    logging.info("best_dice = %f, best_dice_settings = %s" % (best_dice, str(vars(best_dice_settings))))

    logging.info("accuracy = %f, best_accuracy = %f" % (trainer.val_accuracy_history[-1], best_accuracy))
    if best_accuracy < trainer.val_accuracy_history[-1]:
      best_accuracy = trainer.val_accuracy_history[-1]
      best_accuracy_settings = settings
    logging.info("best_accuracy = %f, best_accuracy_settings = %s" % (best_accuracy, str(vars(best_accuracy_settings))))

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

  fe = FeatureExtractor(ds)

  search_for_best_settings(ds, fe)

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
