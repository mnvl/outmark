#! /usr/bin/python3

import os
import sys
from glob import glob
import logging
import random
import hashlib
import unittest
import gflags
import numpy as np
import scipy.stats
import scipy.misc
import nibabel
import pickle

gflags.DEFINE_string("cardiac_training_image_dir",
                     "/home/mel/datasets/CAP/training-images/", "")
gflags.DEFINE_string("cardiac_training_label_dir",
                     "/home/mel/datasets/CAP/training-labels/", "")
gflags.DEFINE_string("cardiac_image_find", ".nii", "")
gflags.DEFINE_string("cardiac_label_replace", "_seg.nii", "")

gflags.DEFINE_string("cervix_training_image_dir",
                     "/home/mel/datasets/Cervix/RawData/Training/img/", "")
gflags.DEFINE_string("cervix_training_label_dir",
                     "/home/mel/datasets/Cervix/RawData/Training/label/", "")
gflags.DEFINE_string("cervix_image_find", "Image", "")
gflags.DEFINE_string("cervix_label_replace", "Mask", "")

gflags.DEFINE_string("abdomen_training_image_dir",
                     "/home/mel/datasets/Abdomen/RawData/Training/img/", "")
gflags.DEFINE_string("abdomen_training_label_dir",
                     "/home/mel/datasets/Abdomen/RawData/Training/label/", "")
gflags.DEFINE_string("abdomen_image_find", "img", "")
gflags.DEFINE_string("abdomen_label_replace", "label", "")

gflags.DEFINE_string("lits_training_image_dir",
                     "/home/mel/datasets/LiTS/train/volume/", "")
gflags.DEFINE_string("lits_training_label_dir",
                     "/home/mel/datasets/LiTS/train/segmentation/", "")
gflags.DEFINE_string("lits_image_find", "volume-", "")
gflags.DEFINE_string("lits_label_replace", "segmentation-", "")

gflags.DEFINE_string("dataset_cache_dir", "/home/mel/datasets/cache/", "")

gflags.DEFINE_string("dataset", "LiTS", "")

FLAGS = gflags.FLAGS


class DataSet(object):

    def get_size(self):
        raise NotImplementedError

    def get_image_and_label(self, index):
        raise NotImplementedError

    def get_classnames(self):
        raise NotImplementedError

    def get_filenames(self, index):
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

        assert len(self.training_set) > 0, "No images found in dataset dir %s." % image_dir

        self.training_set = sorted(self.training_set)

    def get_size(self):
        return len(self.training_set)

    def get_image_and_label(self, index):
        (image_file, label_file) = self.training_set[index]
        image = nibabel.load(image_file)
        image_data = np.swapaxes(image.get_data(), 0, 2)

        label = nibabel.load(label_file)
        label_data = np.swapaxes(label.get_data(), 0, 2)

        assert image.shape == label.shape

        logging.debug("Read image %s and label %s, shape = %s." %
                      (image_file, label_file, (str(image_data.shape))))
        logging.debug(str(image.header))
        logging.debug(str(label.header))

        return (image_data, label_data)

    def get_filenames(self, index):
        return self.training_set[index]


class CardiacDataSet(BasicDataSet):

    def __init__(self):
        super(CardiacDataSet, self).__init__(
            FLAGS.cardiac_training_image_dir,
            FLAGS.cardiac_image_find,
            FLAGS.cardiac_training_label_dir,
            FLAGS.cardiac_label_replace)

    def get_classnames(self):
        return [
            "background",
            "cardiac",
        ]

    def get_image_and_label(self, index):
        image, label = super(CardiacDataSet, self).get_image_and_label(index)

        # WARNING: this dataset contains time dimension -- we merge it with depth
        # dimension, this operation is harmless as our segmentation algorithm is 2d
        def prepare(x):
            x = np.swapaxes(x, 1, 3)
            x = x.reshape((-1, x.shape[2], x.shape[3]))
            return x

        image = prepare(image)
        label = prepare(label)

        assert image.shape == label.shape
        logging.debug("Preprocessed: shape = %s." % str(image.shape))

        return image, label

class TestCardiacDataSet(unittest.TestCase):

    def test_loading_training_set(self):
        cardiac = CardiacDataSet()
        for index in range(cardiac.get_size()):
            image, label = cardiac.get_image_and_label(index)
            logging.info("Image shape is %s." % str(image.shape))
            assert image.shape == label.shape, image.shape + \
                " != " + label.shape

    def test_calculate_class_frequencies(self):
        cardiac = CardiacDataSet()
        indices = [random.randint(0, cardiac.get_size() - 1)
                   for i in range(10)]
        labels = [cardiac.get_image_and_label(index)[1] for index in indices]
        labels = np.concatenate([l.reshape(-1) for l in labels])
        assert np.unique(labels).shape[0] == len(
            cardiac.get_classnames()), np.unique(labels)
        print(labels.shape)
        freqs = scipy.stats.itemfreq(labels)
        print(freqs)
        print(freqs[:, 0], np.max(freqs[:, 1]) / freqs[:, 1])


class CervixDataSet(BasicDataSet):

    def __init__(self):
        super(CervixDataSet, self).__init__(
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
        super(AbdomenDataSet, self).__init__(
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
        super(LiTSDataSet, self).__init__(
            FLAGS.lits_training_image_dir,
            FLAGS.lits_image_find,
            FLAGS.lits_training_label_dir,
            FLAGS.lits_label_replace)

    def get_classnames(self):
        return [
            "(0) background",
            "(1) liver",
            "(2) lesion",
        ]


class TestLiTSDataSet(unittest.TestCase):

    def test_loading_training_set(self):
        lits = LiTSDataSet()
        index = random.randint(0, lits.get_size() - 1)
        image, label = lits.get_image_and_label(index)
        logging.info("Image shape is %s." % str(image.shape))
        logging.info("Image labels are %s." % np.unique(label))
        assert image.shape == label.shape, image.shape + " != " + label.shape
        assert np.unique(label).shape[0] <= len(
            lits.get_classnames()), np.unique(label)

    def test_calculate_class_frequencies(self):
        lits = LiTSDataSet()
        indices = [random.randint(0, lits.get_size() - 1) for i in range(5)]
        labels = [lits.get_image_and_label(index)[1] for index in indices]
        labels = np.concatenate(labels)
        assert np.unique(labels).shape[0] == len(
            lits.get_classnames()), np.unique(labels)
        print(labels.shape)
        freqs = scipy.stats.itemfreq(labels)
        print(freqs)
        print(freqs[:, 0], freqs[:, 1] / np.max(freqs[:, 1]))


class ScalingDataSet(DataSet):

    def __init__(self, dataset, width_or_height):
        self.dataset = dataset
        self.width_or_height = width_or_height

    def get_size(self):
        return self.dataset.get_size()

    def get_image_and_label(self, index):
        image, label = self.dataset.get_image_and_label(index)

        assert image.shape == label.shape

        label = label.astype(np.uint8)

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

        logging.info("Scaled image from %s to %s." %
                     (str(image.shape), str(new_image.shape)))

        return (new_image, new_label)

    def get_classnames(self):
        return self.dataset.get_classnames()

    def get_filenames(self, index):
        return self.dataset.get_filenames(index)


class ShardingDataSet(DataSet):

    def __init__(self, dataset, shards_per_item, extra_depth):
        self.dataset = dataset
        self.shards_per_item = shards_per_item
        self.extra_depth = extra_depth

    def get_size(self):
        return self.dataset.get_size() * self.shards_per_item

    def get_image_and_label(self, index):
        item_index = index // self.shards_per_item
        shard_index = index % self.shards_per_item

        image, label = self.dataset.get_image_and_label(item_index)
        assert image.shape == label.shape

        D, H, W = image.shape

        depth_per_shard = D // self.shards_per_item

        slice_begin = shard_index * depth_per_shard
        slice_end = slice_begin + depth_per_shard

        slice_begin -= self.extra_depth
        slice_end += self.extra_depth

        if slice_begin < 0:
            slice_begin = 0
        if slice_end > D - 1:
            slice_end = D - 1

        return (image[slice_begin:slice_end, :, :], label[slice_begin:slice_end])

    def get_classnames(self):
        return self.dataset.get_classnames()

    def get_filenames(self, index):
        item_index = index // self.shards_per_item
        shard_index = index % self.shards_per_item
        image_filename, label_filename = self.dataset.get_filenames(item_index)
        postfix = "shard_" + str(shard_index)
        return (image_filename + postfix, label_filename + postfix)


class CachingDataSet(DataSet):

    def __init__(self, dataset, prefix):
        self.dataset = dataset
        self.prefix = prefix

    def get_size(self):
        return self.dataset.get_size()

    def make_cache_filename(self, index):
        (image_filename, label_filename) = self.dataset.get_filenames(index)
        filenames_hash = hashlib.md5(
            (image_filename + ":" + label_filename).encode("utf-8")).hexdigest()
        return "%s/%s_%03d_%s.cache" % (FLAGS.dataset_cache_dir,
                                        self.prefix, index, filenames_hash)

    def get_image_and_label(self, index):
        filename = self.make_cache_filename(index)

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


class MemoryCachingDataSet(DataSet):

    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def get_size(self):
        return self.dataset.get_size()

    def get_image_and_label(self, index):
        if index in self.cache:
            return self.cache[index]
        image_and_label = self.dataset.get_image_and_label(index)
        self.cache[index] = image_and_label
        return image_and_label

    def get_classnames(self):
        return self.dataset.get_classnames()


class TestCachingDataSet(unittest.TestCase):

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


def create_dataset():
    if FLAGS.dataset == "Cardiac":
        return ScalingDataSet(CardiacDataSet(), 512)

    if FLAGS.dataset == "Cervix":
        return CervixDataSet()

    if FLAGS.dataset == "Abdomen":
        return AbdomenDataSet()

    if FLAGS.dataset == "LiTS":
        return LiTSDataSet()

    raise NotImplementedError()
