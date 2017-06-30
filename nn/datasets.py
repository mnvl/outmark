#! /usr/bin/python3

import os
import sys
from glob import glob
import logging
import random
import unittest
import gflags
import numpy as np
import scipy.misc
import nibabel
import pickle

FLAGS = gflags.FLAGS

gflags.DEFINE_string("cardiac_training_image_dir",
                     "/home/mel/datasets/Cardiac/averaged-training-images/", "")
gflags.DEFINE_string("cardiac_training_label_dir",
                     "/home/mel/datasets/Cardiac/averaged-training-labels/", "")
gflags.DEFINE_string("cardiac_image_find", ".nii", "")
gflags.DEFINE_string("cardiac_label_replace", "_seg.nii", "")

gflags.DEFINE_string("cervix_training_image_dir",
                     "/home/mel/datasets/Cervix/RegData/img/", "")
gflags.DEFINE_string("cervix_training_label_dir",
                     "/home/mel/datasets/Cervix/RegData/label/", "")
gflags.DEFINE_string("cervix_image_find", "Image", "")
gflags.DEFINE_string("cervix_label_replace", "Mask", "")

gflags.DEFINE_string("abdomen_training_image_dir",
                     "/home/mel/datasets/Abdomen/RawData/Training/img/", "")
gflags.DEFINE_string("abdomen_training_label_dir",
                     "/home/mel/datasets/Abdomen/RawData/Training/label/", "")
gflags.DEFINE_string("abdomen_image_find", "img", "")
gflags.DEFINE_string("abdomen_label_replace", "label", "")

gflags.DEFINE_string("lits_training_image_dir",
                     "/home/mel/datasets/LiTS/train/", "")
gflags.DEFINE_string("lits_training_label_dir",
                     "/home/mel/datasets/LiTS/train/", "")
gflags.DEFINE_string("lits_image_find", "volume-", "")
gflags.DEFINE_string("lits_label_replace", "segmentation-", "")

gflags.DEFINE_string("dataset_cache_dir", "/home/mel/datasets/cache/", "")


class DataSet:

    def get_size(self):
        raise NotImplementedError

    def get_image_and_label(self, index):
        raise NotImplementedError

    def get_classnames(self):
        raise NotImplementedError


class RandomDataSet(DataSet):

    def __init__(self, N=10):
        self.N = N
        self.images = [np.random.uniform(size=(N, 100, 200))
                       for i in range(N)]
        self.labels = [np.random.randint(0, 10, (N, 100, 200))
                       for i in range(N)]

    def get_size(self):
        return self.N

    def get_image_and_label(self, index):
        return (self.images[index], self.labels[index])

    def get_classnames(self):
        return [str(x) for x in range(10)]


class BasicDataSet(DataSet):

    def __init__(self, image_dir, image_find, label_dir, label_replace):
        self.training_set = []

        for image_file in glob(os.path.join(image_dir, "*.nii.gz")):
            logging.info("Found image %s in dataset." % image_file)
            label_file = os.path.join(label_dir,
                                      os.path.basename(
                                          image_file).replace(image_find,
                                                              label_replace))
            self.training_set.append((image_file, label_file))
        assert len(self.training_set) > 0, "No images found in dataset."

        self.training_set = sorted(self.training_set)

    def get_size(self):
        return len(self.training_set)

    def get_image_and_label(self, index):
        (image_file, label_file) = self.training_set[index]
        logging.debug("Reading image %s." % image_file)
        image = nibabel.load(image_file)
        image_data = np.swapaxes(image.get_data(), 0, 2)

        logging.debug("Reading label %s." % label_file)
        label = nibabel.load(label_file)
        label_data = np.swapaxes(label.get_data(), 0, 2)

        return (image_data, label_data)


class CardiacDataSet(BasicDataSet):

    def __init__(self):
        super().__init__(
            FLAGS.cardiac_training_image_dir,
            FLAGS.cardiac_image_find,
            FLAGS.cardiac_training_label_dir,
            FLAGS.cardiac_label_replace)

    def get_classnames(self):
        return [
            "background",
            "cardiac",
        ]


class TestCardiacDataSet(unittest.TestCase):

    def test_loading_training_set(self):
        cardiac = CardiacDataSet()
        for index in range(cardiac.get_size()):
            image, label = cardiac.get_image_and_label(index)
            logging.info("Image shape is %s." % str(image.shape))
            assert image.shape == label.shape, image.shape + \
                " != " + label.shape


class CervixDataSet(BasicDataSet):

    def __init__(self):
        super().__init__(
            FLAGS.cervix_training_image_dir,
            FLAGS.cervix_image_find,
            FLAGS.cervix_training_label_dir,
            FLAGS.cervix_label_replace)

    def get_classnames(self):
        return [
            "(0) background",
          "(1) bladder",
          "(2) uterus",
          "(3) rectum",
          "(4) small bowel",
        ]


class TestCervixDataSet(unittest.TestCase):

    def test_loading_training_set(self):
        cervix = CervixDataSet()
        image, label = cervix.get_image_and_label(0)
        logging.info("Image shape is %s." % str(image.shape))
        assert image.shape == label.shape, image.shape + " != " + label.shape
        assert (np.equal(np.unique(label), np.arange(len(cervix.get_classnames())))).all(), str(
            np.unique(label))


class AbdomenDataSet(BasicDataSet):

    def __init__(self):
        super().__init__(
            FLAGS.abdomen_training_image_dir,
            FLAGS.abdomen_image_find,
            FLAGS.abdomen_training_label_dir,
            FLAGS.abdomen_label_replace)

    def get_classnames(self):
        return [
            "(0) none",
            "(1) spleen",
            "(2) right kidney",
            "(3) left kidney",
            "(4) gallbladder",
            "(5) esophagus",
            "(6) liver",
            "(7) stomach",
            "(8) aorta",
            "(9) inferior vena cava",
            "(10) portal vein and splenic vein",
            "(11) pancreas",
            "(12) right adrenal gland",
            "(13) left adrenal gland",
        ]


class TestAbdomenDataSet(unittest.TestCase):

    def test_loading_training_set(self):
        abdomen = AbdomenDataSet()
        index = random.randint(0, abdomen.get_size() - 1)
        image, label = abdomen.get_image_and_label(index)
        logging.info("Image shape is %s." % str(image.shape))
        assert image.shape == label.shape, image.shape + " != " + label.shape


class LiTSDataSet(BasicDataSet):

    def __init__(self):
        super().__init__(
            FLAGS.lits_training_image_dir,
            FLAGS.lits_image_find,
            FLAGS.lits_training_label_dir,
            FLAGS.lits_label_replace)

    def get_classnames(self):
        return [
            "(0)",
            "(1)",
            "(2)",
        ]


class TestLiTSDataSet(unittest.TestCase):

    def test_loading_training_set(self):
        lits = LiTSDataSet()
        index = random.randint(0, lits.get_size() - 1)
        image, label = lits.get_image_and_label(index)
        logging.info("Image shape is %s." % str(image.shape))
        logging.info("Image labels are %s." % np.unique(label))
        assert image.shape == label.shape, image.shape + " != " + label.shape
        assert np.unique(label).shape[0] <= len(lits.get_classnames()), np.unique(label)


class ScaleDataSet(DataSet):

    def __init__(self, dataset, width_or_height):
        self.dataset = dataset
        self.width_or_height = width_or_height

    def get_size(self):
        return self.dataset.get_size()

    def get_image_and_label(self, index):
        image, label = self.dataset.get_image_and_label(index)

        assert image.shape == label.shape

        D, H, W = image.shape

        if W <= H:
            H = H * self.width_or_height // W
            W = self.width_or_height
        else:
            W = W * self.width_or_height // H
            H = self.width_or_height

        # usually images have very low resolution on vertical axis, so we keep all
        # vertical slices
        new_image = np.zeros((D, H, W))
        new_label = np.zeros((D, H, W), dtype=np.uint8)

        for d in range(D):
            new_image[d, :, :] = scipy.misc.imresize(
                image[d, :, :], (H, W), "bilinear")
            new_label[d, :, :] = scipy.misc.imresize(
                label[d, :, :], (H, W), "nearest")

        return (new_image, new_label)

    def get_classnames(self):
        return self.dataset.get_classnames()


class DataSetCache(DataSet):

    def __init__(self, dataset, prefix):
        self.dataset = dataset
        self.prefix = prefix

    def get_size(self):
        return self.dataset.get_size()

    def get_filename(self, index):
        return "%s/%s_%03d.cache" % (FLAGS.dataset_cache_dir, self.prefix, index)

    def get_image_and_label(self, index):
        filename = self.get_filename(index)

        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
        else:
            image_and_label = self.dataset.get_image_and_label(index)
            with open(filename, "wb") as f:
                pickle.dump(image_and_label, f)
            return image_and_label

    def get_classnames(self):
        return self.dataset.get_classnames()


class TestDataSetCache(unittest.TestCase):

    def test_loading_training_set(self):
        cardiac = CachingDataSet(CardiacDataSet(), prefix="test")
        index = random.randint(0, cardiac.get_size() - 1)
        image, label = cardiac.get_image_and_label(index)
        logging.info("Image shape is %s." % str(image.shape))
        assert image.shape == label.shape, image.shape + " != " + label.shape

if __name__ == '__main__':
    FLAGS(sys.argv)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename='/dev/stderr',
                        filemode='w')
    unittest.main()
