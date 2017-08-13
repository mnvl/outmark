#! /usr/bin/python3

import os
import sys
import random
import numpy as np
import logging
import unittest
import gflags
import json
from timeit import default_timer as timer
import tensorflow as tf
from scipy import misc
import pickle
import util

gflags.DEFINE_boolean(
    "quiet_feature_extractor", False, "")
gflags.DEFINE_string(
    "data_info_json", "/home/mel/datasets/LiTS-baked/info.json", "")
gflags.DEFINE_integer("validation_set_portion", 10, "")

FLAGS = gflags.FLAGS


class FeatureExtractor:

    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

        self.basedir = os.path.dirname(FLAGS.data_info_json)
        with open(FLAGS.data_info_json, "rt") as f:
            self.info_json = json.load(f)

        self.size = self.info_json["size"]
        self.validation_set = []
        self.training_set = []

        for i in range(self.size):
            if i % FLAGS.validation_set_portion == 0:
                self.validation_set.append(i)
            else:
                self.training_set.append(i)

        logging.info("Training set: (%d) %s." %
                     (len(self.training_set), self.training_set))
        logging.info("Validation set: (%d) %s." %
                     (len(self.validation_set), self.validation_set))

        self.good_training_set_slices = []
        for i in self.training_set:
            class_table = self.info_json[str(i)]["class_table"]

            good_slices = 0
            for k, v in class_table.items():
                if k != "0":
                    good_slices += len(v)
                    for j in v:
                        self.good_training_set_slices.append((i, j))

            logging.info("Item %d has %d good slices." % (i, good_slices))

        self.validation_set_slices = []
        for i in self.validation_set:
            for k, v in self.info_json[str(i)]["slices"].items():
                self.validation_set_slices.append((i, int(k)))

        logging.info("Total %d good slices." %
                     (len(self.good_training_set_slices),))

    def augment_image(self, image, label):
        image = image.astype(np.float32)
        label = label.astype(np.uint8)

        random_rotation = random.random() * 20.0 - 10.0
        image = misc.imrotate(image, random_rotation, "bilinear")
        label = misc.imrotate(label, random_rotation, "nearest")

        random_resize = (int(image.shape[0] * (random.random() * 0.2 + 0.9)),
                         int(image.shape[1] * (random.random() * 0.2 + 0.9)))
        image = misc.imresize(image, random_resize, "bilinear")
        label = misc.imresize(label, random_resize, "nearest")

        return image, label

    def normalize_image(self, image, label):
        image = image / np.std(image)
        if not FLAGS.quiet_feature_extractor:
            logging.info(str(np.unique(label)) +
                         "\n" + util.crappyhist(image, bins=20))
        return image, label

    def crop_image_training(self, image, label):
        xs, ys = np.nonzero(label)

        if xs.shape[0] == 0:
            return self.crop_image_validation(image, label)

        i = random.randint(0, xs.shape[0] - 1)
        x1 = xs[i] + random.randint(-self.image_width//2, self.image_width//2)
        y1 = ys[i] + random.randint(-self.image_height//2, self.image_height//2)

        x1 = max(x1 - self.image_width // 2, 0)
        y1 = max(y1 - self.image_height // 2, 0)
        x2 = min(x1 + self.image_width, image.shape[0] - 1)
        y2 = min(y1 + self.image_height, image.shape[1] - 1)

        if x2 - x1 < self.image_width:
            if x1 == 0:
                x2 = x1 + self.image_width
            if x2 == image.shape[0] - 1:
                x1 = x2 - self.image_width

        if y2 - y1 < self.image_height:
            if y1 == 0:
                y2 = y1 + self.image_height
            if y2 == image.shape[1] - 1:
                y1 = y2 - self.image_height

        image = image[x1:x2, y1:y2]
        label = label[x1:x2, y1:y2]

        return image, label

    def crop_image_validation(self, image, label):
        x1 = random.randint(0, image.shape[0] - self.image_width - 1)
        y1 = random.randint(0, label.shape[1] - self.image_height - 1)

        x2 = x1 + self.image_width
        y2 = y1 + self.image_height

        image = image[x1:x2, y1:y2]
        label = label[x1:x2, y1:y2]

        return image, label

    def get_random_image_slice(self, image_index, slice_index):
        info = self.info_json[str(image_index)]["slices"][str(slice_index)]
        filepath = os.path.join(self.basedir, info["filename"])

        start = timer()
        with open(filepath, "rb") as f:
            image, label = pickle.load(f)
        end = timer()

        if not FLAGS.quiet_feature_extractor:
            logging.info("Loaded image %d (with shape %s) slice %d from %s in %.3f secs." %
                         (image_index, image.shape, slice_index, filepath, end - start))

        image, label = self.augment_image(image, label)
        image, label = self.normalize_image(image, label)

        return image, label

    def get_random_training_example(self):
        image_index, slice_index = random.choice(self.good_training_set_slices)
        image, label = self.get_random_image_slice(image_index, slice_index)
        image, label = self.crop_image_training(image, label)
        return image, label

    def get_random_trining_batch(self, batch_size):
        X = np.zeros(
            (batch_size, self.image_height, self.image_width), dtype=np.float32)
        y = np.zeros(
            (batch_size, self.image_height, self.image_width), dtype=np.uint8)
        for i in range(batch_size):
            X[i, :, :], y[i, :, :] = self.get_random_training_example()
        return X, y

    def get_random_validation_example(self):
        image_index, slice_index = random.choice(self.validation_set_slices)
        image, label = self.get_random_image_slice(image_index, slice_index)
        image, label = self.crop_image_validation(image, label)
        return image, label

    def get_random_validation_batch(self, batch_size):
        X = np.zeros(
            (batch_size, self.image_height, self.image_width), dtype=np.float32)
        y = np.zeros(
            (batch_size, self.image_height, self.image_width), dtype=np.uint8)
        for i in range(batch_size):
            X[i, :, :], y[i, :, :] = self.get_random_validation_example()
        return X, y

    def get_validation_set_size(self):
        return len(self.validation_set)

    def get_validation_set_item(self, index):
        image_index = self.validation_set[index]

        info = self.info_json[str(image_index)]["whole"]
        filepath = os.path.join(self.basedir, info["filename"])

        start = timer()
        with open(filepath, "rb") as f:
            image, label = pickle.load(f)
        end = timer()

        logging.info("Loaded image %d (with shape %s) from %s in %.3f secs." %
                     (image_index, image.shape, filepath, end - start))

        image, label = self.normalize_image(image, label)

        return image, label

    def get_validation_set_items(self):
        images = []
        labels = []
        for i in range(self.get_validation_set_size()):
            image, label = self.get_validation_set_item(i)
            images.append(image)
            labels.append(label)
        return images, labels

    def get_classnames(self):
        return self.info_json["classnames"]

    def get_num_classes(self):
        return len(self.get_classnames())


class TestFeatureExtractor(unittest.TestCase):

    def test_basic(self):
        fe = FeatureExtractor(256, 256)

        for i in range(10):
            image, label = fe.get_random_training_example()
            misc.imsave("test_%d_image.png" % i, image)
            misc.imsave("test_%d_label.png" %
                        i, label * (250 // fe.get_num_classes()))

        for i in range(5):
            image, label = fe.get_random_validation_example()
            misc.imsave("valid_%d_image.png" % i, image)
            misc.imsave("valid_%d_label.png" %
                        i, label * (250 // fe.get_num_classes()))

        fe.get_validation_set_item(0)

    def test_batch(self):
        fe = FeatureExtractor(256, 256)
        fe.get_random_trining_batch(10)
        fe.get_random_validation_batch(10)

if __name__ == '__main__':
    FLAGS(sys.argv)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename='/dev/stderr',
                        filemode='w')
    unittest.main()
