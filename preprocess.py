#! /usr/bin/python3

import os
import sys
import random
import numpy as np
import logging
import unittest
import gflags
import json
from PIL import Image
from timeit import default_timer as timer
from scipy import misc
import multiprocessing as mp
import util

gflags.DEFINE_boolean(
    "verbose_feature_extractor", True, "")
gflags.DEFINE_string(
    "data_info_json", "/home/mel/datasets/LiTS-baked/info.json", "")
gflags.DEFINE_integer("validation_set_portion", 10, "")

FLAGS = gflags.FLAGS


class FeatureExtractor:

    def __init__(self, image_width, image_height, batch_size):
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size

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
        self.all_training_set_slices = []
        self.good_validation_set_slices = []
        self.all_validation_set_slices = []
        for i in range(self.size):
            is_training_set = (i in set(self.training_set))

            class_table = self.info_json[str(i)]["class_table"]

            good_slices = 0
            bad_slices = 0
            for k, v in class_table.items():
                if k != "0":
                    good_slices += len(v)
                    if is_training_set:
                        for j in v:
                            self.good_training_set_slices.append((i, j))
                            self.all_training_set_slices.append((i, j))
                    else:
                        for j in v:
                            self.good_validation_set_slices.append((i, j))
                            self.all_validation_set_slices.append((i, j))
                else:
                    bad_slices += len(v)
                    if is_training_set:
                        for j in v:
                            self.all_training_set_slices.append((i, j))
                    else:
                        for j in v:
                            self.all_validation_set_slices.append((i, j))

            logging.info("Item %d has %d good/%d bad slices." %
                         (i, good_slices, bad_slices))

        logging.info("%d/%d good slices in training set." %
                     (len(self.good_training_set_slices), len(self.all_training_set_slices)))
        logging.info("%d/%d good slices in validation set." %
                     (len(self.good_validation_set_slices), len(self.all_validation_set_slices)))

    def augment_image(self, image, label):
        # rotate fills an image with zeros
        shift = np.min(image.reshape(-1))
        image -= shift

        image = Image.fromarray(image, mode="F")
        label = Image.fromarray(label, mode="P")

        random_rotation = random.random() * 20.0 - 10.0
        image = image.rotate(random_rotation, resample=Image.BILINEAR)
        label = label.rotate(random_rotation, resample=Image.NEAREST)

        random_resize = (int(image.size[0] * (random.random() * 0.2 + 0.9)),
                         int(image.size[1] * (random.random() * 0.2 + 0.9)))
        image = image.resize(random_resize, resample=Image.BILINEAR)
        label = label.resize(random_resize, resample=Image.NEAREST)

        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.uint8)

        image += shift

        if FLAGS.verbose_feature_extractor:
            logging.info(str(np.unique(label)) +
                         "\n" + util.text_hist(image, bins=20))

        return image, label

    def normalize_image(self, image, label):
        image = image.astype(np.float32)
        label = label.astype(np.uint8)

        if FLAGS.verbose_feature_extractor:
            logging.info(str(np.unique(label)) +
                         "\n" + util.text_hist(image, bins=20))

        return image, label

    def crop_image_smart(self, image, label):
        xs, ys = np.nonzero(label)

        if xs.shape[0] == 0:
            return self.crop_image_random(image, label)

        i = random.randint(0, xs.shape[0] - 1)
        x1 = xs[i] + random.randint(
            -self.image_width // 2, self.image_width // 2)
        y1 = ys[i] + random.randint(
            -self.image_height // 2, self.image_height // 2)

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

    def crop_image_random(self, image, label):
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
        image, label = util.read_image_and_label(filepath)
        end = timer()

        if FLAGS.verbose_feature_extractor:
            logging.info("Loaded image %d (with shape %s) slice %d from %s in %.3f secs." %
                         (image_index, image.shape, slice_index, filepath, end - start))

        image, label = self.normalize_image(image, label)
        image, label = self.augment_image(image, label)

        return image, label

    def get_random_example(self, all_slices, good_slices):
        if random.choice([False, True]):
            image_index, slice_index = random.choice(all_slices)
            image, label = self.get_random_image_slice(
                image_index, slice_index)
            image, label = self.crop_image_random(image, label)
        else:
            image_index, slice_index = random.choice(good_slices)
            image, label = self.get_random_image_slice(
                image_index, slice_index)
            image, label = self.crop_image_smart(image, label)

        return image, label

    def get_random_training_example(self):
        return self.get_random_example(self.all_training_set_slices, self.good_training_set_slices)

    def get_random_training_batch(self):
        X = np.zeros(
            (self.batch_size, self.image_height, self.image_width), dtype=np.float32)
        y = np.zeros(
            (self.batch_size, self.image_height, self.image_width), dtype=np.uint8)
        for i in range(self.batch_size):
            X[i, :, :], y[i, :, :] = self.get_random_training_example()
        return X, y

    def get_random_validation_example(self):
        return self.get_random_example(self.all_validation_set_slices, self.good_validation_set_slices)

    def get_random_validation_batch(self):
        X = np.zeros(
            (self.batch_size, self.image_height, self.image_width), dtype=np.float32)
        y = np.zeros(
            (self.batch_size, self.image_height, self.image_width), dtype=np.uint8)
        for i in range(self.batch_size):
            X[i, :, :], y[i, :, :] = self.get_random_validation_example()
        return X, y

    def get_validation_set_size(self):
        return len(self.validation_set)

    def get_validation_set_item(self, index):
        image_index = self.validation_set[index]

        info = self.info_json[str(image_index)]["whole"]
        filepath = os.path.join(self.basedir, info["filename"])

        start = timer()
        image, label = util.read_image_and_label(filepath)
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
        fe = FeatureExtractor(256, 256, 10)

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
        fe = FeatureExtractor(256, 256, 10)
        fe.get_random_training_batch()
        fe.get_random_validation_batch()


class FeatureExtractorProcess:

    def __init__(self, image_width, image_height, batch_size):
        self.queue_train = mp.Queue(maxsize=1)
        self.queue_valid = mp.Queue(maxsize=1)
        self.process = mp.Process(
            target=FeatureExtractorProcess.work,
            args=(self.queue_train, self.queue_valid, image_width, image_height, batch_size))
        self.process.start()

    def __del__(self):
        self.stop()

    def get_random_training_batch(self):
        return self.queue_train.get()

    def stop(self):
        self.process.terminate()

    def work(queue_train, queue_valid, image_width, image_height, batch_size):
        fe = FeatureExtractor(image_width, image_height, batch_size)
        while True:
            if queue_valid.empty():
                batch = fe.get_random_validation_batch()
                queue_valid.put(batch)

            batch = fe.get_random_training_batch()
            queue_train.put(batch)


class TestFeatureExtractorProcess(unittest.TestCase):

    def test_basic(self):
        fe = FeatureExtractorProcess(256, 256, 10)

        for i in range(10):
            fe.get_random_training_batch()

        fe.stop()

if __name__ == '__main__':
    FLAGS(sys.argv)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename='/dev/stderr',
                        filemode='w')
    unittest.main()
