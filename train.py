#! /usr/bin/python3

import os
import sys
import time
import itertools
import datetime
import logging
import random
import pickle
import numpy as np
import tensorflow as tf
import scipy.misc
import gflags
from timeit import default_timer as timer
from vnet import VNet
from preprocess import FeatureExtractor
import util

gflags.DEFINE_boolean("notebook", False, "")
gflags.DEFINE_integer("num_steps", 500, "")
gflags.DEFINE_integer("batch_size", 8, "")
gflags.DEFINE_integer("image_depth", 16, "")
gflags.DEFINE_integer("image_height", 160, "")
gflags.DEFINE_integer("image_width", 160, "")
gflags.DEFINE_integer("shards_per_item", 10, "")
gflags.DEFINE_string("dataset", "LiTS", "")
gflags.DEFINE_string("output", "./output/", "")
gflags.DEFINE_string("mode", "train", "{hypersearch, train}")
gflags.DEFINE_string("read_model", "", "")
gflags.DEFINE_integer("estimate_every_steps", 25, "")
gflags.DEFINE_integer("validate_every_steps", 500, "")

FLAGS = gflags.FLAGS


class Trainer:

    def __init__(self, settings, feature_extractor):
        self.feature_extractor = feature_extractor

        self.S = settings
        self.model = VNet(settings)
        self.model.add_layers()
        self.model.add_optimizer()

        self.val_accuracy_history = []
        self.val_iou_history = []

        self.step = 0
        self.model.start()

    def read_model(self, filepath):
        self.model.read(filepath + "tf")

        if os.path.isfile(filepath + "vars"):
            with open(filepath + "vars", "rb") as f:
                data = pickle.load(f)
                self.step = data.get("step")

    def write_model(self, filepath):
        self.model.write(filepath + "tf")

        with open(filepath + "vars", "wb") as f:
            data = {
                "step": self.step,
            }
            pickle.dump(data, f)

    def train(self, num_steps):
        validate_every_steps = min(num_steps, FLAGS.validate_every_steps)

        start_time = time.time()

        val_estimate_accuracy_history = []
        val_estimate_iou_history = []

        while self.step < num_steps:
            (X, y) = self.feature_extractor.get_random_trining_batch(
                self.S.batch_size)
            X = np.expand_dims(X, 1)
            y = np.expand_dims(y, 1)

            (loss, train_accuracy, train_iou) = self.model.fit(X, y, self.step)

            eta = int((time.time() - start_time) / (
                self.step + 1) * (num_steps - self.step))
            eta = str(datetime.timedelta(seconds=eta))

            if (self.step + 1) % validate_every_steps == 0:
                (val_accuracy, val_iou) = self.validate_full()

                self.val_accuracy_history.append(val_accuracy)
                self.val_iou_history.append(val_iou)

                logging.info("[step %6d/%6d, eta = %s] accuracy = %f, iou = %f, loss = %f, val_accuracy = %f, val_iou = %f" %
                             (self.step, num_steps, eta, train_accuracy, train_iou, loss, val_accuracy, val_iou))
            elif (self.step + 1) % FLAGS.estimate_every_steps == 0:
                (val_accuracy, val_iou) = self.validate_fast()

                val_estimate_accuracy_history = val_estimate_accuracy_history[
                    -100:] + [val_accuracy]
                val_estimate_iou_history = val_estimate_iou_history[
                    -100:] + [val_iou]

                val_accuracy_estimate = np.mean(val_estimate_accuracy_history)
                val_iou_estimate = np.mean(val_estimate_iou_history)

                logging.info("[step %6d/%6d, eta = %s] accuracy = %f, iou = %f, loss = %f, val_accuracy_estimate = %f, val_iou_estimate = %f" %
                             (self.step, num_steps, eta, train_accuracy, train_iou, loss, val_accuracy_estimate, val_iou_estimate))
            else:
                logging.info("[step %6d/%6d, eta = %s] accuracy = %f, iou = %f, loss = %f" %
                             (self.step, num_steps, eta, train_accuracy, train_iou, loss))

            self.step += 1

    def validate_fast(self):
        X_val, y_val = self.feature_extractor.get_random_validation_batch(
            self.S.batch_size)
        X_val = np.expand_dims(X_val, 1)
        y_val = np.expand_dims(y_val, 1)

        y_pred = self.model.predict(X_val)
        val_accuracy = util.accuracy(y_pred, y_val)
        val_iou = util.iou(y_pred, y_val, self.S.num_classes)
        return (val_accuracy, val_iou)

    def validate_full(self):
        # these are just lists of images as they can have mismatching depth
        # dimensions
        (val_images, val_labels) = fe.get_validation_set_items()

        # debug code:
        #i, l = fe.get_validation_set_item(0)
        #val_images = [i]
        #val_labels = [l]

        pred_labels = []
        for i, (X_val, y_val) in enumerate(zip(val_images, val_labels)):
            start = timer()
            y_pred = self.model.segment(X_val)
            pred_labels.append(y_pred)
            end = timer()

            logging.info("Segmented image %d with shape %s in %.3f secs." %
                         (i, X_val.shape, end - start))

        self.write_images(pred_labels, val_images, val_labels)
        self.write_model(FLAGS.output + "/checkpoint_%06d." % self.step)

        pred_labels_flat = np.concatenate(
            [x.flatten() for x in pred_labels])
        val_labels_flat = np.concatenate(
            [x.flatten() for x in val_labels])

        val_accuracy = util.accuracy(pred_labels_flat, val_labels_flat)
        val_iou = util.iou(
            pred_labels_flat, val_labels_flat, self.S.num_classes)

        return (val_accuracy, val_iou)

    def write_images(self, pred, val_images, val_labels):
        image = val_images
        label = val_labels

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
            FLAGS.output + "/%06d_1_eq.png" % self.step,
            (label == pred).astype(np.uint8) * 250)
        scipy.misc.imsave(
            FLAGS.output + "/%06d_2_pred.png" % self.step,
            pred.astype(np.uint8) * (250 / self.feature_extractor.get_num_classes()))
        scipy.misc.imsave(
            FLAGS.output + "/%06d_3_label.png" % self.step,
            label.astype(np.uint8) * (250 / self.feature_extractor.get_num_classes()))
        scipy.misc.imsave(FLAGS.output + "/%06d_4_mask.png" % self.step, mask)
        scipy.misc.imsave(FLAGS.output + "/%06d_5_mix.png" %
                          self.step, (100. + np.expand_dims(image, 2)) * (1. + mask))

    def clear(self):
        self.model.stop()


def get_validation_set_size(ds):
    size = (ds.get_size() // FLAGS.shards_per_item) // 5
    if size > 20:
        size = 20
    # should be multiple of FLAGS.shards_per_item so the image would not leak
    size = size * FLAGS.shards_per_item
    return size


def make_basic_settings(fe, hypersearch=False):
    s = VNet.Settings()
    s.batch_size = FLAGS.batch_size
    s.loss = random.choice(["softmax", "iou"])
    s.num_classes = fe.get_num_classes()
    s.class_weights = random.choice(([1.0, 3.0, 8.0], [1.0, 1.0, 1.0]))
    s.image_depth = FLAGS.image_depth
    s.image_height = FLAGS.image_width
    s.image_width = FLAGS.image_height
    s.keep_prob = random.uniform(0.5, 1.0) if hypersearch else 0.7
    s.l2_reg = 1.0e-05 * ((10 ** random.uniform(-3, 3)) if hypersearch else 1)
    s.learning_rate = 1.0e-04 * \
        ((10 ** random.uniform(-3, 3)) if hypersearch else 1)
    s.num_conv_blocks = 4
    s.num_conv_channels = 40
    s.num_dense_channels = 0
    s.num_dense_layers = 1
    s.use_batch_norm = random.choice([True, False]) if hypersearch else False
    return s


def make_best_settings_for_dataset(vanilla=False):
    if FLAGS.dataset == "Cardiac":
        # *** dice = 0.73
        # s = VNet.Settings()
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

        s = VNet.Settings()
        s.batch_size = FLAGS.batch_size
        s.class_weights = [1.0, 1.0]
        s.image_depth = 1
        s.image_height = 224
        s.image_width = 224
        s.keep_prob = 0.85
        s.l2_reg = 0.0003
        s.learning_rate = 7.65e-05 * 0.01
        s.loss = "iou"
        s.num_classes = 2
        s.num_conv_blocks = 3
        s.num_conv_channels = 30
        s.num_dense_channels = 0
        s.num_dense_layers = 1
        s.use_batch_norm = False
        return s
    elif FLAGS.dataset == "LiTS":
        # best_iou = 0.048529,
        # best_iou_settings = {'loss': 'iou', 'num_dense_channels': 0, 'class_weights': [1.0, 1.0, 1.0], 'num_conv_channels': 30, 'keep_prob': 0.6796631579428167, 'image_width': 224, 'image_depth': 16, 'num_conv_blocks': 3, 'image_height': 224, 'batch_size': 1, 'use_batch_norm': False, 'num_dense_layers': 1, 'learning_rate': 2.1335824070750984e-05, 'num_classes': 3, 'l2_reg': 3.5827450874760806e-06}
        # best_accuracy = 0.980120
        # best_accuracy_settings = {'loss': 'iou', 'num_dense_channels': 0,
        # 'class_weights': [1.0, 1.0, 1.0], 'num_conv_channels': 30,
        # 'keep_prob': 0.9966614841201717, 'image_width': 224, 'image_depth':
        # 16, 'num_conv_blocks': 3, 'image_height': 224, 'batch_size': 1,
        # 'use_batch_norm': False, 'num_dense_layers': 1, 'learning_rate':
        # 4.9802527145240384e-05, 'num_classes': 3, 'l2_reg':
        # 3.430971119758406e-05}
        s = VNet.Settings()
        s.loss = "iou"
        s.batch_size = FLAGS.batch_size
        s.class_weights = [1.0, 3.0, 8.0]
        assert FLAGS.image_depth == 1
        s.image_depth = 1
        assert FLAGS.image_height == 448
        s.image_height = 448
        assert FLAGS.image_width == 448
        s.image_width = 448
        s.keep_prob = 0.7
        s.learning_rate = 1.1742089305437838e-05
        s.l2_reg = 1.1534199440443374e-05
        s.num_classes = 3
        s.num_conv_blocks = 4
        s.num_conv_channels = 20
        s.num_dense_channels = 0
        s.num_dense_layers = 1
        s.use_batch_norm = True
        return s
    else:
        raise "Unknown dataset"


def search_for_best_settings(fe):
    best_iou = -1
    best_iou_settings = None
    best_accuracy = -1
    best_accuracy_settings = None

    for i in range(100):
        settings = make_basic_settings(fe, hypersearch=True)

        logging.info("*** try %d, settings: %s" % (i, str(vars(settings))))

        try:
            trainer = Trainer(settings, fe)
            trainer.train(FLAGS.num_steps)
        except tf.errors.ResourceExhaustedError as e:
            logging.info("Resource exhausted: %s", e.message)
            trainer.clear()
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


def train_model(fe):
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

    fe = FeatureExtractor(FLAGS.image_width, FLAGS.image_height)

    if FLAGS.mode == "hypersearch":
        search_for_best_settings(fe)
    elif FLAGS.mode == "train":
        train_model(fe)
    else:
        raise "Unknown mode " + FLAGS.mode
