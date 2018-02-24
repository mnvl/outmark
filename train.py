#! /usr/bin/python3

import os
import sys
import time
import itertools
import datetime
import logging
import random
import pickle
import json
import numpy as np
import tensorflow as tf
import scipy.misc
import gflags
import hyperopt
from timeit import default_timer as timer
from vnet import VNet
from preprocess import FeatureExtractor
import image_server
import metrics
import util

gflags.DEFINE_boolean("notebook", False, "")
gflags.DEFINE_integer("num_steps", 500, "")
gflags.DEFINE_integer("batch_size", 8, "")
gflags.DEFINE_integer("image_depth", 16, "")
gflags.DEFINE_integer("image_height", 160, "")
gflags.DEFINE_integer("image_width", 160, "")
gflags.DEFINE_string("settings", "LiTS", "")
gflags.DEFINE_string("output", "./output/", "")
gflags.DEFINE_string("mode", "train", "{hyperopt, train, export}")
gflags.DEFINE_string("read_model", "", "")
gflags.DEFINE_string("export_model", "", "")
gflags.DEFINE_integer("estimate_every_steps", 25, "")
gflags.DEFINE_integer("validate_every_steps", 2000, "")

FLAGS = gflags.FLAGS


class Trainer:

    def __init__(self, settings):
        self.S = settings

        self.feature_extractor = FeatureExtractor(
            self.S.image_width, self.S.image_height, self.S.batch_size)

        self.model = VNet(settings)
        self.model.add_layers()
        self.model.add_optimizer()

        self.history = util.AttributeDict()
        self.history.val_accuracy = []
        self.history.val_iou = []
        self.history.train_loss = []
        self.history.train_accuracy = []
        self.history.train_iou = []
        self.history.val_loss_estimate = []
        self.history.val_accuracy_estimate = []
        self.history.val_iou_estimate = []

        self.step = 0
        self.model.start()

    def read_model(self, filepath):
        self.model.read(filepath + "tf")

        if os.path.isfile(filepath + "vars"):
            try:
                with open(filepath + "vars", "rb") as f:
                    data = pickle.load(f)
            except:
                with open(filepath + "vars", "rt") as f:
                    data = json.load(f)

            self.step = data.get("step") + 1

            history = data.get("history", None)
            if history is not None:
                self.history = util.AttributeDict(history)

    def write_model(self, filepath):
        self.model.write(filepath + "tf")

        data = {
            "step": self.step,
            "history": self.history,
        }

        with open(filepath + "vars", "wt") as f:
            json.dump(data, f)

    def export_model(self, filepath):
        self.model.export(filepath)

    def train(self, num_steps):
        validate_every_steps = min(num_steps, FLAGS.validate_every_steps)

        start_time = time.time()

        while self.step < num_steps:
            (X, y) = self.feature_extractor.get_random_training_batch()
            X = np.expand_dims(X, 1)
            y = np.expand_dims(y, 1)

            (loss, y_pred, train_accuracy, train_iou) = self.model.fit(X, y, self.step)
            self.write_images(y_pred[0], X[0], y[0], text = "train")

            eta = int((time.time() - start_time) / (
                self.step + 1) * (num_steps - self.step))
            eta = str(datetime.timedelta(seconds=eta))

            if (self.step + 1) % validate_every_steps == 0:
                (val_accuracy, val_iou) = self.validate_full()

                self.history.val_accuracy.append(float(val_accuracy))
                self.history.val_iou.append(float(val_iou))

                logging.info("[step %6d/%6d, eta = %s] accuracy = %f, iou = %f, loss = %f, segmentation: val_accuracy = %f, val_iou = %f" %
                             (self.step, num_steps, eta, train_accuracy, train_iou, loss, val_accuracy, val_iou))
            elif (self.step + 1) % FLAGS.estimate_every_steps == 0:
                val_loss, val_accuracy, val_iou = self.validate_fast()

                self.history.train_loss.append(float(loss))
                self.history.train_accuracy.append(float(train_accuracy))
                self.history.train_iou.append(float(train_iou))
                self.history.val_loss_estimate.append(float(val_loss))
                self.history.val_accuracy_estimate.append(float(val_accuracy))
                self.history.val_iou_estimate.append(float(val_iou))

                g2i = image_server.graphs_to_image
                images = [g2i(self.history.train_loss,
                              self.history.val_loss_estimate,
                              title = "loss",),
                          g2i(self.history.train_accuracy,
                              self.history.val_accuracy_estimate,
                              title = "accuracy"),
                          g2i(self.history.train_iou,
                              self.history.val_iou_estimate,
                              title = "iou"),
                          g2i(self.history.val_accuracy,
                              title = "segment accuracy",
                              moving_average = False),
                          g2i(self.history.val_iou,
                              title = "segment iou",
                              moving_average = False)]
                image_server.put_images("graph", images, keep_only_last = True)

                logging.info("[step %6d/%6d, eta = %s] accuracy = %f, iou = %f, loss = %f, estimation: val_accuracy = %f, val_iou = %f, val_loss = %f" %
                             (self.step, num_steps, eta, train_accuracy, train_iou, loss, val_accuracy, val_iou, val_loss))
            else:
                logging.info("[step %6d/%6d, eta = %s] accuracy = %f, iou = %f, loss = %f" %
                             (self.step, num_steps, eta, train_accuracy, train_iou, loss))

            self.step += 1

    def validate_fast(self):
        X_val, y_val = self.feature_extractor.get_random_validation_batch()
        X_val = np.expand_dims(X_val, 1)
        y_val = np.expand_dims(y_val, 1)

        val_loss, y_val_pred, val_accuracy, val_iou = self.model.predict(X_val, y_val)

        for i in range(X_val.shape[0]):
            self.write_images(y_val_pred[i], X_val[i], y_val[i], text = "validate")

        return val_loss, val_accuracy, val_iou

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
        val_accuracy = []
        val_iou = []
        for i, (X_val, y_val) in enumerate(zip(val_images, val_labels)):
            start = timer()
            y_val_pred = self.model.segment(X_val)
            end = timer()

            pred_labels.append(y_val_pred)
            val_accuracy.append(metrics.accuracy(y_val, y_val_pred))
            val_iou.append(metrics.iou(y_val, y_val_pred, self.feature_extractor.get_num_classes()))

            logging.info("Segmented image %d with shape %s in %.3f secs." %
                         (i, X_val.shape, end - start))

        self.write_images(pred_labels, val_images, val_labels, text= "segment", save_to_disk = True)
        self.write_model(FLAGS.output + "/checkpoint_%06d." % self.step)

        val_accuracy = np.mean(val_accuracy)
        val_iou = np.mean(val_iou)

        return (val_accuracy, val_iou)

    def write_images(self, pred, image, label, text = "", save_to_disk = False):
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

        eq = (label == pred).astype(np.uint8) * 250
        pred = pred.astype(np.uint8) * (250 // self.feature_extractor.get_num_classes())
        label = label.astype(np.uint8) * (250 // self.feature_extractor.get_num_classes())
        mask = np.dstack((eq, label, pred))

        image_server.put_images(text, (image, label, pred, eq, mask))

        if not save_to_disk:
            return

        scipy.misc.imsave(
            FLAGS.output + "/%06d_0_image.png" % self.step, image)
        scipy.misc.imsave(
            FLAGS.output + "/%06d_1_eq.png" % self.step,
            eq)
        scipy.misc.imsave(
            FLAGS.output + "/%06d_2_pred.png" % self.step,
            pred)
        scipy.misc.imsave(
            FLAGS.output + "/%06d_3_label.png" % self.step,
            label)
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
    s = VNet.Settings()

    s.loss = "softmax"
    s.batch_size = FLAGS.batch_size
    s.image_depth = FLAGS.image_depth
    s.image_height = FLAGS.image_width
    s.image_width = FLAGS.image_height
    s.keep_prob = 0.5
    s.learning_rate = 0.05
    s.num_conv_blocks = 4
    s.num_conv_channels = 64
    s.l2_reg = 0.0
    s.use_batch_norm = False
    s.num_dense_layers = 0

    if FLAGS.settings == "Abdomen":
        s.num_classes = 13
        s.class_weights = [1.0] + [5.0]*12
    elif FLAGS.settings == "Cardiac":
        s.num_classes = 2
        s.class_weights = [1.0, 5.0]
    elif FLAGS.settings == "LiTS":
        s.num_classes = 3
        s.class_weights = [1.0, 5.0, 5.0]
        s.l2_reg = 1.0e-6
    elif FLAGS.settings == "LCTSC":
        s.learning_rate = 0.0005
        s.num_classes = 6
        s.class_weights = [1.0] + [5.0]*5
        s.keep_prob = 0.5
        s.l2_reg = 1.0e-4
    elif FLAGS.settings == "tissue":
        s.learning_rate = 0.0001
        s.num_classes = 6
        s.class_weights = [1.0] + [5.0]*5
        s.keep_prob = 0.5
        s.l2_reg = 1.0e-3
    else:
        raise ValueError("Unknown dataset")

    return s

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
    s.num_conv_blocks = 5
    s.num_conv_channels = 40
    s.num_dense_layers = 1
    s.use_batch_norm = False

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

def export_model():
    settings = make_best_settings()
    trainer = Trainer(settings)
    trainer.read_model(FLAGS.read_model)
    trainer.export_model(FLAGS.export_model)


if __name__ == '__main__':
    FLAGS(sys.argv)

    try:
        util.setup_logging()
        image_server.start()

        if FLAGS.mode == "hyperopt":
            search_for_best_settings()
        elif FLAGS.mode == "train":
            train_model()
        elif FLAGS.mode == "export":
            export_model()
        else:
            raise ValueError("Unknown mode " + FLAGS.mode)
    finally:
        image_server.stop()
