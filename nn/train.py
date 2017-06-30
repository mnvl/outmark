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
from volunet import VolUNet
from datasets import DataSetCache, ScaleDataSet, CardiacDataSet, CervixDataSet, AbdomenDataSet, LiTSDataSet
from preprocess import FeatureExtractor
import util

gflags.DEFINE_boolean("notebook", False, "")
gflags.DEFINE_string("dataset", "Cervix", "")
gflags.DEFINE_integer("num_steps", 500, "")
gflags.DEFINE_integer("image_depth", 16, "")
gflags.DEFINE_integer("image_height", 160, "")
gflags.DEFINE_integer("image_width", 160, "")
gflags.DEFINE_string("output", "./output/", "")
gflags.DEFINE_string("mode", "train", "{fiddle, train}")
gflags.DEFINE_string("read_model", "", "")

FLAGS = gflags.FLAGS


class Trainer:

    def __init__(self, settings, dataset, validation_set_size, feature_extractor):
        self.dataset = dataset
        self.training_set_size = dataset.get_size() - validation_set_size
        self.validation_set_size = validation_set_size
        self.feature_extractor = feature_extractor

        self.S = settings
        self.model = VolUNet(settings)
        self.model.add_layers()
        self.model.add_optimizer()

        self.train_loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.val_iou_history = []

        self.dataset_shuffle = np.arange(dataset.get_size())

        saved_seed = np.random.seed()
        np.random.seed(1)
        np.random.shuffle(self.dataset_shuffle)
        np.random.seed(saved_seed)

        self.step = 0

        self.model.start()

    def read_model(self, filepath):
        self.model.read(filepath + "tf")

        if os.path.isfile(filepath + "vars"):
            with open(filepath + "vars", "rb") as f:
                data = pickle.load(f)
                self.dataset_shuffle = data.get("dataset_shuffle")
                self.step = data.get("step")

    def write_model(self, filepath):
        self.model.write(filepath + "tf")

        with open(filepath + "vars", "wb") as f:
            data = {
                "dataset_shuffle": self.dataset_shuffle,
                "step": self.step,
            }
            pickle.dump(data, f)

    def train(self, num_steps, estimate_every_steps=20, validate_every_steps=500):
        val_accuracy_estimate = 0
        val_iou_estimate = 0

        validate_every_steps = min(num_steps, validate_every_steps)

        # these are just lists of images as they can have mismatching depth
        # dimensions
        logging.info("loading validation set")
        (self.val_images, self.val_labels) = fe.get_images(
            self.dataset_shuffle[
                np.arange(self.training_set_size, self.dataset.get_size())],
          self.S.image_height, self.S.image_width)

        start_time = time.time()

        while self.step < num_steps:
            (X, y) = fe.get_examples(
                self.dataset_shuffle[
                    np.random.randint(
                        0, self.training_set_size - 1, self.S.batch_size)],
              self.S.image_depth, self.S.image_height, self.S.image_width)
            X = np.expand_dims(X, axis=4)

            (loss, train_accuracy, train_iou) = self.model.fit(X, y, self.step)

            self.train_loss_history.append(loss)
            self.train_accuracy_history.append(train_accuracy)

            eta = int((time.time() - start_time) / (
                self.step + 1) * (num_steps - self.step))
            eta = str(datetime.timedelta(seconds=eta))

            if (self.step + 1) % validate_every_steps == 0:
                (val_accuracy, val_iou) = self.validate_full()

                logging.info("[step %6d/%6d, eta = %s] accuracy = %f, iou = %f, loss = %f, val_accuracy = %f, val_iou = %f" %
                             (self.step, num_steps, eta, train_accuracy, train_iou, loss, val_accuracy, val_iou))

                if self.step == 0:
                    val_accuracy_estimate = val_accuracy
                    val_iou_estimate = val_iou

                self.val_accuracy_history.append(val_accuracy)
                self.val_iou_history.append(val_iou)
            elif (self.step + 1) % estimate_every_steps == 0:
                (val_accuracy, val_iou) = self.validate_fast()

                val_accuracy_estimate = val_accuracy_estimate * \
                    0.8 + val_accuracy * 0.2
                val_iou_estimate = val_iou_estimate * 0.8 + val_iou * 0.2

                logging.info("[step %6d/%6d, eta = %s] accuracy = %f, iou = %f, loss = %f, val_accuracy_estimate = %f, val_iou_estimate = %f" %
                             (self.step, num_steps, eta, train_accuracy, train_iou, loss, val_accuracy_estimate, val_iou_estimate))
            else:
                logging.info("[step %6d/%6d, eta = %s] accuracy = %f, iou = %f, loss = %f" %
                             (self.step, num_steps, eta, train_accuracy, train_iou, loss))

            self.step += 1

    def validate_fast(self):
        (X_val, y_val) = fe.get_examples(
            self.dataset_shuffle[
                np.random.randint(
                    self.training_set_size, self.dataset.get_size(
                    ) - 1, self.S.batch_size)],
            self.S.image_depth, self.S.image_height, self.S.image_width)
        X_val = np.expand_dims(X_val, axis=4)

        y_pred = self.model.predict(X_val)

        val_accuracy = util.accuracy(y_pred, y_val)
        val_iou = util.iou(y_pred, y_val)

        return (val_accuracy, val_iou)

    def validate_full(self):
        pred_labels = []
        for i, (X_val, y_val) in enumerate(zip(self.val_images, self.val_labels)):
            X_val = np.expand_dims(X_val, axis=4)
            y_pred = self.model.segment_image(X_val)
            pred_labels.append(y_pred)

        self.write_images(pred_labels)
        self.write_model(FLAGS.output + "/checkpoint_%06d." % self.step)

        pred_labels_flat = np.concatenate(
            [x.flatten() for x in pred_labels])
        val_labels_flat = np.concatenate(
            [x.flatten() for x in self.val_labels])

        val_accuracy = util.accuracy(pred_labels_flat, val_labels_flat)
        val_iou = util.iou(pred_labels_flat, val_labels_flat)

        return (val_accuracy, val_iou)

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

        scipy.misc.imsave(
            FLAGS.output + "/%06d_0_image.png" % self.step, image)
        scipy.misc.imsave(
            FLAGS.output + "/%06d_1_eq.png" % self.step, (label == pred))
        scipy.misc.imsave(FLAGS.output + "/%06d_2_pred.png" % self.step, pred)
        scipy.misc.imsave(
            FLAGS.output + "/%06d_3_label.png" % self.step, label)
        scipy.misc.imsave(FLAGS.output + "/%06d_4_mask.png" % self.step, mask)
        scipy.misc.imsave(FLAGS.output + "/%06d_5_mix.png" %
                          self.step, (100. + np.expand_dims(image, 2)) * (1. + mask))

    def clear(self):
        self.model.stop()


def get_validation_set_size(ds):
    size = ds.get_size() // 5
    if size > 20:
        size = 20
    return size


def make_basic_settings(fiddle=False):
    settings = VolUNet.Settings()
    settings.batch_size = 1
    settings.loss = "iou" # random.choice(["softmax", "iou"])
    settings.num_classes = len(ds.get_classnames())
    settings.class_weights = [1., 90.]
    settings.image_depth = FLAGS.image_depth
    settings.image_height = FLAGS.image_width
    settings.image_width = FLAGS.image_height
    settings.keep_prob = random.uniform(0.7, 0.9) if fiddle else 0.84
    settings.l2_reg = 1.0e-6 * ((10 ** random.uniform(-3, 3)) if fiddle else 1)
    settings.learning_rate = 1.0e-5 * ((10 ** random.uniform(-3, 3)) if fiddle else 1)
    settings.num_conv_blocks = 3
    settings.num_conv_channels = 20
    settings.num_dense_channels = 0
    settings.num_dense_layers = 1
    settings.use_batch_norm = random.choice([True, False]) if fiddle else False
    return settings


def make_best_settings_for_dataset(vanilla=False):
    if FLAGS.dataset == "Cardiac":
        # *** dice = 0.73
        # s = VolUNet.Settings()
        # s.batch_size = 5
        # s.class_weights = [1, 28.0268060324304]
        # s.image_depth = 1
        # s.image_height = 224
        # s.image_width = 224
        # s.keep_prob = 0.8383480946442744
        # s.l2_reg = 3.544580353901791e-05
        # s.learning_rate = 0.0003604126178497249 * 0.1
        # s.num_classes = 2
        # s.num_conv_blocks = 3
        # s.num_conv_channels = 30
        # s.num_dense_channels = 0
        # s.num_dense_layers = 1
        # s.use_batch_norm = False
        # return s

        s = VolUNet.Settings()
        s.batch_size = 5
        s.class_weights = [1, 90.]
        s.image_depth = 1
        s.image_height = 224
        s.image_width = 224
        s.keep_prob = 0.8306712895560182
        s.l2_reg = 6.904512516575549e-05
        s.learning_rate = 0.00023104616030970466
        s.num_classes = 2
        s.num_conv_blocks = 4
        s.num_conv_channels = 50
        s.num_dense_channels = 0
        s.num_dense_layers = 1
        s.use_batch_norm = False
        return s
    elif FLAGS.dataset == "LiTS":
        s = VolUNet.Settings()
        s.batch_size =  1
        s.class_weights =  [1., 30., 30.]
        s.image_depth =  24
        s.image_height =  160
        s.image_width =  160
        s.keep_prob =  0.77
        s.l2_reg =  6.1e-07
        s.learning_rate =  3.56e-05
        s.num_classes =  3
        s.num_conv_blocks =  3
        s.num_conv_channels =  20
        s.num_dense_channels =  0
        s.num_dense_layers =  1
        s.use_batch_norm =  False
        return s
    else:
        raise "Unknown dataset"


def search_for_best_settings(ds, fe):
    best_iou = -1
    best_iou_settings = None
    best_accuracy = -1
    best_accuracy_settings = None

    for i in range(100):
        settings = make_basic_settings(fiddle=True)

        logging.info("*** try %d, settings: %s" % (i, str(vars(settings))))

        try:
            trainer = Trainer(settings, ds, get_validation_set_size(ds), fe)
            trainer.train(FLAGS.num_steps)
        except tf.errors.ResourceExhaustedError as e:
            trainer.clear()
            logging.info("Resource exhausted: %s", e.message)
            continue
        finally:
            trainer.clear()

        logging.info("iou = %f, best_iou = %f" %
                     (trainer.val_iou_history[-1], best_iou))
        if best_iou < trainer.val_iou_history[-1]:
            best_iou = trainer.val_iou_history[-1]
            best_iou_settings = settings
        logging.info("best_iou = %f, best_iou_settings = %s" %
                     (best_iou, str(vars(best_iou_settings))))

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
    trainer = Trainer(settings, ds, get_validation_set_size(ds), fe)
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
    elif FLAGS.dataset == "Cervix":
        ds = CervixDataSet()
    elif FLAGS.dataset == "Abdomen":
        ds = AbdomenDataSet()
    elif FLAGS.dataset == "LiTS":
        ds = LiTSDataSet()
    else:
        print("Unknown dataset: %s" % FLAGS.dataset, file=sys.stderr)
        sys.exit(1)

    ds = ScaleDataSet(ds, min(FLAGS.image_width, FLAGS.image_height))

    ds = DataSetCache(ds, prefix="%s_%dx%d" % (FLAGS.dataset,
                                               FLAGS.image_height,
                                               FLAGS.image_width))
    fe = FeatureExtractor(ds)

    if FLAGS.mode == "fiddle":
        search_for_best_settings(ds, fe)
    elif FLAGS.mode == "train":
        train_model(ds, fe)
    else:
        raise "Unknown mode " + FLAGS.mode
