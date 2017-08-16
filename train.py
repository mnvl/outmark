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
import hyperopt
from timeit import default_timer as timer
from vnet import VNet
from preprocess import FeatureExtractor, FeatureExtractorProcess
import util

gflags.DEFINE_boolean("notebook", False, "")
gflags.DEFINE_integer("num_steps", 500, "")
gflags.DEFINE_integer("batch_size", 8, "")
gflags.DEFINE_integer("image_depth", 16, "")
gflags.DEFINE_integer("image_height", 160, "")
gflags.DEFINE_integer("image_width", 160, "")
gflags.DEFINE_integer("shards_per_item", 10, "")
gflags.DEFINE_string("settings", "LiTS", "")
gflags.DEFINE_string("output", "./output/", "")
gflags.DEFINE_string("mode", "train", "{hyperopt, train}")
gflags.DEFINE_string("read_model", "", "")
gflags.DEFINE_integer("estimate_every_steps", 25, "")
gflags.DEFINE_integer("validate_every_steps", 500, "")

FLAGS = gflags.FLAGS


class Trainer:

    def __init__(self, settings):
        self.S = settings

        self.feature_extractor = FeatureExtractor(
            self.S.image_width, self.S.image_height, self.S.batch_size)
        self.feature_extractor_process = self.feature_extractor
        # self.feature_extractor_process = FeatureExtractorProcess(
        #     self.S.image_width, self.S.image_height, self.S.batch_size)

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
            (X, y) = self.feature_extractor_process.get_random_training_batch()
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

                logging.info("[step %6d/%6d, eta = %s] accuracy = %f, iou = %f, loss = %f, segmentation: val_accuracy = %f, val_iou = %f" %
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
        X_val, y_val = self.feature_extractor_process.get_random_validation_batch()
        X_val = np.expand_dims(X_val, 1)
        y_val = np.expand_dims(y_val, 1)

        y_pred = self.model.predict(X_val)

        self.write_images(y_pred[0], X_val[0], y_val[0])

        val_accuracy = util.accuracy(y_pred, y_val)
        val_iou = util.iou(y_pred, y_val, self.S.num_classes)

        return (val_accuracy, val_iou)

    def validate_full(self):
        # these are just lists of images as they can have mismatching depth
        # dimensions
        if True:
            (val_images, val_labels) = self.feature_extractor.get_validation_set_items()
        else:
            i, l = self.feature_extractor.get_validation_set_item(0)
            val_images = [i]
            val_labels = [l]

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

    def write_images(self, pred, image, label):
        if isinstance(image, list):
            i = random.randint(0, len(image) - 1)
            image = image[i]
            label = label[i]
            pred = pred[i]

        if image.shape[0] > 1:
            j = random.randint(0, image.shape[0] - 1)
            image = image[j,:,:]
            label = label[j,:,:]
            pred = pred[j,:,:]
        else:
            image = image[0,:,:]
            label = label[0,:,:]
            pred = pred[0,:,:]

        mask = np.dstack((label == pred, label, pred)).astype(np.float32)

        scipy.misc.imsave(
            FLAGS.output + "/%06d_0_image.png" % self.step, image)
        scipy.misc.imsave(
            FLAGS.output + "/%06d_1_eq.png" % self.step,
            (label == pred).astype(np.uint8) * 250)
        scipy.misc.imsave(
            FLAGS.output + "/%06d_2_pred.png" % self.step,
            pred.astype(np.uint8) * (250 // self.feature_extractor.get_num_classes()))
        scipy.misc.imsave(
            FLAGS.output + "/%06d_3_label.png" % self.step,
            label.astype(np.uint8) * (250 // self.feature_extractor.get_num_classes()))
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


def make_best_settings():
    if FLAGS.settings == "Cardiac":
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
    elif FLAGS.settings == "LiTS":
        # {'class_weights': (1.0, 87.11018068138196, 82.22532812532825), 'keep_prob': 0.7597034105999735, 'l2_reg': 0.018348912433853164, 'learning_rate': 0.023574000050660678, 'loss': 'softmax', 'use_batch_norm': True}
        s = VNet.Settings()
        s.batch_size = FLAGS.batch_size
        s.loss = "softmax"
        s.num_classes = 3
        s.class_weights = [1.0, 90.0, 90.0]
        s.image_depth = FLAGS.image_depth
        s.image_height = FLAGS.image_width
        s.image_width = FLAGS.image_height
        s.keep_prob = 0.75
        s.l2_reg = 0.0183
        s.learning_rate = 0.0235
        s.num_conv_blocks = 4
        s.num_conv_channels = 40
        s.num_dense_channels = 0
        s.num_dense_layers = 1
        s.use_batch_norm = True
        return s
    else:
        raise "Unknown dataset"


def train_and_calculate_metric(params):
    logging.info("params = " + str(params))

    s = VNet.Settings()
    s.batch_size = FLAGS.batch_size
    s.loss = params["loss"]
    s.num_classes = len(params["class_weights"])
    s.class_weights = params["class_weights"]
    s.image_depth = FLAGS.image_depth
    s.image_height = FLAGS.image_width
    s.image_width = FLAGS.image_height
    s.keep_prob = params["keep_prob"]
    s.l2_reg = params["l2_reg"]
    s.learning_rate = params["learning_rate"]
    s.num_conv_blocks = 4
    s.num_conv_channels = 40
    s.num_dense_channels = 0
    s.num_dense_layers = 1
    s.use_batch_norm = params["use_batch_norm"]

    try:
        trainer = Trainer(s)
        trainer.train(FLAGS.num_steps)
    except tf.errors.ResourceExhaustedError as e:
        logging.info("Resource exhausted: %s", e.message)
        trainer.clear()
        return {'status': hyperopt.STATUS_FAIL}
    finally:
        trainer.clear()

    logging.info(
        "iou = " + str(trainer.val_iou_history[-1]) + ", params = " + str(params))

    return {'loss': -trainer.val_iou_history[-1], 'status': hyperopt.STATUS_OK}


def search_for_best_settings():
    objective = train_and_calculate_metric
    num_classes = FeatureExtractor(1, 1, 1).get_num_classes()

    hp = hyperopt.hp

    space = {
        "loss": hp.choice("loss", ["softmax", "iou"]),
        "class_weights": [1.0] + [hp.uniform("class_weight_%d" % i, 1.0, 100.0) for i in range(1, num_classes)],
        "keep_prob": hp.uniform("keep_prob", 0.5, 1.0),
        "l2_reg": hp.uniform("l2_norm", 1.0e-6, 0.1),
        "learning_rate": hp.uniform("learning_rate", 1.0e-6, 0.1),
        "use_batch_norm": hp.choice("use_batch_norm", [False, True]),
    }

    best = hyperopt.fmin(
        objective, space, algo=hyperopt.tpe.suggest, max_evals=25)

    logging.info("best settings: " + str(best))


def train_model():
    settings = make_best_settings()
    logging.info("Settings: " + str(settings))
    trainer = Trainer(settings)
    if len(FLAGS.read_model) > 0:
        trainer.read_model(FLAGS.read_model)
    trainer.train(FLAGS.num_steps)


if __name__ == '__main__':
    FLAGS(sys.argv)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename='/dev/stderr',
                        filemode='w')

    if FLAGS.mode == "hyperopt":
        search_for_best_settings()
    elif FLAGS.mode == "train":
        train_model()
    else:
        raise "Unknown mode " + FLAGS.mode
