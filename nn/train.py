#! /usr/bin/python3

import os
import sys
import time
import datetime
import logging
import random
import pickle
import numpy as np
import tensorflow as tf
import scipy.misc
import gflags
from unet import UNet
from dense_unet import DenseUNet
from datasets import CachingDataSet, CardiacDataSet, CervixDataSet, AbdomenDataSet
from preprocess import FeatureExtractor
import util

gflags.DEFINE_boolean("notebook", False, "")
gflags.DEFINE_string("dataset", "Cardiac", "")
gflags.DEFINE_integer("num_steps", 500, "")
gflags.DEFINE_string("output", "./output/", "")
gflags.DEFINE_string("mode", "train", "{fiddle, train}")
gflags.DEFINE_string("read_model", "", "")

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

    def read_model(self, filepath):
        self.model.read(filepath + "tf")

        if os.path.isfile(filepath + "vars"):
            with open(filepath + "vars", "rb") as f:
                data = pickle.load(f)
                self.dataset_shuffle = data.get("dataset_shuffle")

    def write_model(self, filepath):
        self.model.write(filepath + "tf")

        with open(filepath + "vars", "wb") as f:
            data = {
                "dataset_shuffle": self.dataset_shuffle,
            }
            pickle.dump(data, f)

    def train(self, num_steps, estimate_every_steps=20, validate_every_steps=100):
        val_accuracy_estimate = 0
        val_dice_estimate = 0

        # these are just lists of images as they can have mismatching depth
        # dimensions
        logging.info("loading validation set")
        (self.val_images, self.val_labels) = fe.get_images(
            self.dataset_shuffle[
                np.arange(self.training_set_size, self.dataset.get_size())],
          self.S.image_height, self.S.image_width)

        start_time = time.time()

        for self.step in range(num_steps):
            (X, y) = fe.get_examples(
                self.dataset_shuffle[
                    np.random.randint(
                        0, self.training_set_size - 1, self.S.batch_size)],
              self.S.image_depth, self.S.image_height, self.S.image_width)
            X = np.expand_dims(X, axis=4)

            (loss, train_accuracy, train_dice) = self.model.fit(X, y, self.step)

            self.train_loss_history.append(loss)
            self.train_accuracy_history.append(train_accuracy)

            eta = int((time.time() - start_time) / (self.step + 1) * (num_steps - self.step))
            eta = str(datetime.timedelta(seconds = eta))

            if (self.step + 1) % validate_every_steps == 0 or self.step == 0:
                (val_accuracy, val_dice) = self.validate_full()

                logging.info("[step %6d/%6d, eta = %s] accuracy = %f, dice = %f, loss = %f, val_accuracy = %f, val_dice = %f" %
                             (self.step, num_steps, eta, train_accuracy, train_dice, loss, val_accuracy, val_dice))

                if self.step == 0:
                    val_accuracy_estimate = val_accuracy
                    val_dice_estimate = val_dice

                self.val_accuracy_history.append(val_accuracy)
                self.val_dice_history.append(val_dice)
            elif (self.step + 1) % estimate_every_steps == 0:
                (val_accuracy, val_dice) = self.validate_fast()

                val_accuracy_estimate = val_accuracy_estimate * 0.8 + val_accuracy * 0.2
                val_dice_estimate = val_dice_estimate * 0.8 + val_dice * 0.2


                logging.info("[step %6d/%6d, eta = %s] accuracy = %f, dice = %f, loss = %f, val_accuracy_estimate = %f, val_dice_estimate = %f" %
                             (self.step, num_steps, eta, train_accuracy, train_dice, loss, val_accuracy_estimate, val_dice_estimate))
            else:
                logging.info("[step %6d/%6d, eta = %s] accuracy = %f, dice = %f, loss = %f" %
                             (self.step, num_steps, eta, train_accuracy, train_dice, loss))

    def validate_fast(self):
        (X_val, y_val) = fe.get_examples(
            self.dataset_shuffle[
                np.random.randint(
                    self.training_set_size, self.dataset.get_size() - 1, self.S.batch_size)],
            self.S.image_depth, self.S.image_height, self.S.image_width)
        X_val = np.expand_dims(X_val, axis=4)

        y_pred = self.model.predict(X_val)

        val_accuracy = util.accuracy(y_pred, y_val)
        val_dice = util.dice(y_pred, y_val)

        return (val_accuracy, val_dice)

    def validate_full(self):
        pred_labels = []
        for i, (X_val, y_val) in enumerate(zip(self.val_images, self.val_labels)):
            X_val = np.expand_dims(X_val, axis=4)
            y_pred = self.model.segment_image(X_val)
            pred_labels.append(y_pred)

        if FLAGS.mode != "fiddle":
            self.write_images(pred_labels)
            self.write_model(FLAGS.output + "/checkpoint_%06d." % self.step)

        pred_labels_flat = np.concatenate(
            [x.flatten() for x in pred_labels])
        val_labels_flat = np.concatenate(
            [x.flatten() for x in self.val_labels])

        val_accuracy = util.accuracy(pred_labels_flat, val_labels_flat)
        val_dice = util.dice(pred_labels_flat, val_labels_flat)

        return (val_accuracy, val_dice)

    def write_images(self, pred):
        image = self.val_images
        label = self.val_labels

        i = random.randint(0, len(image) - 1)
        image = image[i]
        label = label[i]
        pred = pred[i]

        j = random.randint(0, image.shape[0] - 1)
        image = image[j, :, :]
        label = label[j, :, :]
        pred = pred[j, :, :]

        mask = np.dstack((label == pred, label, pred))

        pred = pred.astype(np.float32)
        label = label.astype(np.float32)
        mask = mask.astype(np.float32)

        scipy.misc.imsave(FLAGS.output + "/%06d_0_image.png" % self.step, image)
        scipy.misc.imsave(
            FLAGS.output + "/%06d_1_eq.png" % self.step, (label == pred))
        scipy.misc.imsave(FLAGS.output + "/%06d_2_pred.png" % self.step, pred)
        scipy.misc.imsave(FLAGS.output + "/%06d_3_label.png" % self.step, label)
        scipy.misc.imsave(FLAGS.output + "/%06d_4_mask.png" % self.step, mask)
        scipy.misc.imsave(FLAGS.output + "/%06d_5_mix.png" %
                          self.step, (100. + np.expand_dims(image, 2)) * (1. + mask))

    def clear(self):
        self.model.stop()


def make_basic_settings(fiddle=False):
    settings = UNet.Settings()
    settings.batch_size = 5
    settings.class_weights = [1] + [random.uniform(25., 31.) if fiddle else 28.] * (settings.num_classes - 1)
    settings.image_depth = random.choice([1]) if fiddle else 1
    settings.image_height = 64 if FLAGS.notebook else 224
    settings.image_width = 64 if FLAGS.notebook else 224
    settings.keep_prob = random.uniform(0.7, 0.9) if fiddle else 0.84
    settings.l2_reg = 3.5e-05 * ((10 ** random.uniform(-1, 1)) if fiddle else 1)
    settings.learning_rate = 3.54e-05 * ((10 ** random.uniform(-1, 1)) if fiddle else 1)
    settings.num_classes = len(ds.get_classnames())
    settings.num_conv_blocks = 4 #random.randint(2, 4) if fiddle else 2
    settings.num_conv_channels = 50 #random.randint(30, 90) if fiddle else 110
    settings.num_conv_layers_per_block = 2 #random.randint(2, 3) if fiddle else 2
    settings.num_dense_channels = 0 #random.randint(90, 130) if fiddle else 128
    settings.num_dense_layers = 1 #random.randint(1, 2) if fiddle else 1
    settings.use_batch_norm = random.choice([True, False]) if fiddle else False
    return settings


def make_best_settings_for_dataset(vanilla = False):
    if FLAGS.dataset == 'Cardiac':
        settings = UNet.Settings()
        settings.batch_size = 5
        settings.class_weights = [1, 28.0268060324304]
        settings.image_depth = 1
        settings.image_height = 224
        settings.image_width = 224
        settings.keep_prob = 0.8383480946442744
        settings.l2_reg = 3.544580353901791e-05
        settings.learning_rate = 0.0003604126178497249 * 0.1
        settings.num_classes = 2
        settings.num_conv_blocks = 3
        settings.num_conv_channels = 30
        settings.num_conv_layers_per_block = 2
        settings.num_dense_channels = 0
        settings.num_dense_layers = 1
        settings.use_batch_norm = False
        return settings
    else:
        raise "Unknown dataset"


def search_for_best_settings(ds, fe):
    best_dice = -1
    best_dice_settings = None
    best_accuracy = -1
    best_accuracy_settings = None

    for i in range(100):
        settings = make_basic_settings(fiddle=True)

        logging.info("*** try %d, settings: %s" % (i, str(vars(settings))))

        try:
            trainer = Trainer(settings, ds, 4 * ds.get_size() // 5, fe)
            trainer.train(FLAGS.num_steps)
        except tf.errors.ResourceExhaustedError as e:
            trainer.clear()
            logging.info("Resource exhausted: %s", e.message)
            continue
        finally:
            trainer.clear()

        logging.info("dice = %f, best_dice = %f" %
                     (trainer.val_dice_history[-1], best_dice))
        if best_dice < trainer.val_dice_history[-1]:
            best_dice = trainer.val_dice_history[-1]
            best_dice_settings = settings
        logging.info("best_dice = %f, best_dice_settings = %s" %
                     (best_dice, str(vars(best_dice_settings))))

        logging.info("accuracy = %f, best_accuracy = %f" %
                     (trainer.val_accuracy_history[-1], best_accuracy))
        if best_accuracy < trainer.val_accuracy_history[-1]:
            best_accuracy = trainer.val_accuracy_history[-1]
            best_accuracy_settings = settings
        logging.info("best_accuracy = %f, best_accuracy_settings = %s" %
                     (best_accuracy, str(vars(best_accuracy_settings))))


def train_model(ds, fe):
    settings = make_best_settings_for_dataset()
    logging.info("Settings: " + str(settings))
    trainer = Trainer(settings, ds, 4 * ds.get_size() // 5, fe)
    if len(FLAGS.read_model) > 0:
        trainer.read_model(FLAGS.read_model)
    trainer.train(FLAGS.num_steps)


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
        print("Unknown dataset: %s" % FLAGS.dataset, file=sys.stderr)
        sys.exit(1)

    fe = FeatureExtractor(ds)

    if FLAGS.mode == "fiddle":
        search_for_best_settings(ds, fe)
    elif FLAGS.mode == "train":
        train_model(ds, fe)
    else:
        raise "Unknown mode " + FLAGS.mode
