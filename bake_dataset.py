#! /usr/bin/python3

import os
import sys
import gflags
import datasets
import logging
import pickle
import json
import numpy as np

gflags.DEFINE_integer("process_first", -1, "")
gflags.DEFINE_integer("image_height", 224, "")
gflags.DEFINE_integer("image_width", 224, "")
gflags.DEFINE_string("output_dir", "", "")

FLAGS = gflags.FLAGS

def build_class_table(label):
    unique_labels = np.unique(label)
    logging.info("Unique labels: %s." % (str(unique_labels)))
    assert np.unique(label).shape[0] * 2 > unique_labels[-1], "Suspicious labels"

    table = {}
    for z in range(label.shape[0]):
        for l in np.unique(label[z, :, :]):
            l = str(l)
            if l in table:
                table[l].append(z)
            else:
                table[l] = [z]

    logging.info("Classes and frequences: %s." % (str({x:len(y) for x, y in table.items()})))

    return table

def process(ds, index):
    image, label = ds.get_image_and_label(index)

    assert image.shape == label.shape

    info = {
        "class_table": build_class_table(label),
    }

    return image, label, info

def main():
    ds = datasets.create_dataset()
    ds = datasets.ScalingDataSet(ds, max(FLAGS.image_height, FLAGS.image_width))

    info_table = {}

    for index in range(ds.get_size()):
        if FLAGS.process_first > 0 and index + 1 == FLAGS.process_first: break

        logging.info("Processing %d/%d, %.1f%% completed..." % (index, ds.get_size(), float(index) / ds.get_size() * 100))
        image, label, info = process(ds, index)

        image_filename, label_filename = ds.get_filenames(index)
        image_filename = os.path.basename(image_filename)
        label_filename = os.path.basename(label_filename)

        info["image_filename"] = image_filename
        info["label_filename"] = label_filename

        info_table[str(index)] = info

        record = (image, label, info_table)

        filename = os.path.join(FLAGS.output_dir, ("%d.pickle" % index))
        logging.info("Writing image, label, and info to %s." % filename)
        with open(filename, "wb") as f:
            pickle.dump(record, f)

    filename = os.path.join(FLAGS.output_dir, "info.json")
    logging.info("Writing info to %s." % filename)
    with open(filename, "wt") as f:
        json.dump(info_table, f, indent=4, sort_keys = True)

if __name__ == '__main__':
    FLAGS(sys.argv)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename='/dev/stderr',
                        filemode='w')

    main()
