#! /usr/bin/python3

import os
import sys
import gflags
import logging
import json
import numpy as np
import string
import datasets
import util
import multiprocessing

gflags.DEFINE_integer("process_first", 1000, "")
gflags.DEFINE_string("output_dir", "", "")

FLAGS = gflags.FLAGS


def build_class_table(label):
    unique_labels = np.unique(label)
    logging.info("Unique labels: %s." % (str(unique_labels)))
    assert (np.unique(label).shape[0] * 2 >
            unique_labels[-1]), "Suspicious labels"

    table = {}
    for z in range(label.shape[0]):
        for l in np.unique(label[z, :, :]):
            l = str(l)
            if l in table:
                table[l].append(z)
            else:
                table[l] = [z]

    logging.info("Classes and frequences: %s." %
                 (str({x: len(y) for x, y in table.items()})))

    return table


def build_slices(index, image, label):
    table = {}
    for z in range(image.shape[0]):
        filename = "%03d_%03d.npz" % (index, z)
        filepath = os.path.join(FLAGS.output_dir, filename)

        logging.info("Writing slice %d/%d: %s." %
                     (z, image.shape[0], filepath))

        util.write_image_and_label(filepath, image[z, :, :], label[z, :, :])

        table[str(z)] = {"filename": filename}
    return table


def build_whole(index, image, label):
    filename = "%03d.npz" % (index,)
    filepath = os.path.join(FLAGS.output_dir, filename)

    logging.info("Writing whole image: %s." % (filepath, ))
    util.write_image_and_label(filepath, image, label)

    table = {"filename": filename}
    return table


def process(arguments):
    ds, index = arguments

    logging.info("Starting %d/%d." % (index, ds.get_size()))

    image, label = ds.get_image_and_label(index)
    logging.debug("Original dtypes: %s (%s .. %s), %s." %
                  (image.dtype, np.min(image), np.max(image), label.dtype))
    logging.debug(util.text_hist(image))

    assert image.shape == label.shape, str(
        image.shape) + " != " + str(label.shape)
    assert image.dtype == np.float32, str(image.dtype)

    label = label.astype(np.uint8)

    info = {
        "class_table": build_class_table(label),
        "whole": build_whole(index, image, label),
        "slices": build_slices(index, image, label),
    }

    logging.info("Finished %d/%d." % (index, ds.get_size()))

    return info


def main():
    ds = datasets.create_dataset()

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    pool = multiprocessing.Pool(processes=16)
    inputs = [(ds, index)
              for index in range(min(FLAGS.process_first, ds.get_size()))]
    results = pool.map(process, inputs)

    info_table = {}
    info_table["classnames"] = ds.get_classnames()
    info_table["size"] = ds.get_size()

    for index, info in enumerate(results):
        image_filename, label_filename = ds.get_filenames(index)
        image_filename = os.path.basename(image_filename)
        label_filename = os.path.basename(label_filename)

        info["image_filename"] = image_filename
        info["label_filename"] = label_filename

        info_table[str(index)] = info

    filename = os.path.join(FLAGS.output_dir, "info.json")
    logging.info("Writing info to %s." % filename)
    with open(filename, "wt") as f:
        json.dump(info_table, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    FLAGS(sys.argv)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename='/dev/stderr',
                        filemode='w')

    main()
