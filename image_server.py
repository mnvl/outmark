#! /usr/bin/python3

import io
import os
import sys
import time
import gflags
import logging
import unittest
import numpy as np
import matplotlib.pyplot as plt
from http.server import BaseHTTPRequestHandler, HTTPServer
from PIL import Image
from threading import Thread, Lock
import hashlib
import util

gflags.DEFINE_integer("image_server_port", 17171, "")
gflags.DEFINE_integer("image_server_rows_per_page", 5, "")
gflags.DEFINE_integer("image_server_storage_per_page", 1000, "")

FLAGS = gflags.FLAGS

_server = None
_key_generator = 1
_table = {}
_images = {}
_queue = {}
_lock = Lock()
_thread = None


class RequestHandler(BaseHTTPRequestHandler):

    def handle_index(self, page=None):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        global _lock
        with _lock:
            if page is None:
                page = list(sorted(_table.keys()))[0]

            text = "<meta http-equiv=\"refresh\" content=\"5\" />"

            text += "<center><table><tr>"
            for p in sorted(_table.keys()):
                p2 = "<font face=\"arial\" size=\"%d\">%s</font>" % (
                    6 if p == page else 4, p)
                p2 = ("<a href=\"/index/%s\">%s</a>" %
                      (p, p2)) if p != page else p2
                text += "<td>%s</td>" % p2
            text += "</tr></table>"

            text += "<table>"
            for images in _table[page]:
                text += "<tr>"
                size = "width=256 height=256" if len(images) > 1 else ""
                for image in images:
                    text += ("<td><img src=\"/image/%d\" %s></td>" % (image, size))
                text += "</tr>"
            text += "</table>"

            text += "<p>pid = %d, argv = %s, cwd = %s" % (
                os.getpid(), str(sys.argv), os.getcwd())
            text += "</center>"

        self.wfile.write(text.encode("utf-8"))

    def handle_image(self, key):
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.end_headers()

        global _lock
        with _lock:
            self.wfile.write(_images[key])

    def do_GET(self):
        if self.path.strip() == "/favicon.ico":
            self.send_response(404)
            return

        if self.path.strip() == "/":
            self.handle_index()
            return

        _, command, arg = self.path.split("/")

        if command == "index":
            self.handle_index(arg)

        if command == "image":
            self.handle_image(int(arg))
            return

    def log_message(self, fmt, *args):
        return

    def log_error(self, fmt, *args):
        logging.error("%s %s %s" %
                      (self.address_string(), self.log_date_time_string(), fmt % args))


def start():
    global _server
    _server = HTTPServer(("", FLAGS.image_server_port),
                         RequestHandler)
    global _thread
    _thread = Thread(target=_server.serve_forever)
    _thread.start()


def stop():
    global _server
    _server.shutdown()


def put_images(page, images, keep_only_last=False):
    global _key_generator
    global _lock
    global _queue

    images = list(images)
    for i, image in enumerate(images):
        if isinstance(image, np.ndarray):
            output = io.BytesIO()
            image = Image.fromarray(image)
            image = image.convert("RGB")
            image.save(output, format="png")
            contents = output.getvalue()
            output.close()

            images[i] = contents

    with _lock:
        keys = []
        for i in range(len(images)):
            keys.append(_key_generator)
            _key_generator += 1

        queue = _queue.get(page, []) + keys
        while len(queue) >= FLAGS.image_server_storage_per_page:
            del _images[queue[0]]
            queue = queue[1:]
        _queue[page] = queue

        for k, i in zip(keys, images):
            _images[k] = i

        if keep_only_last:
            # transpose a row for better experience
            _table[page] = [[k] for k in keys]
        else:
            old = _table.get(page, [])
            if len(old) >= FLAGS.image_server_rows_per_page:
                old = old[:-1]
            _table[page] = [keys] + old


def figure_to_image(fig):
    output = io.BytesIO()
    fig.savefig(output, format='png')
    contents = output.getvalue()
    output.close()
    plt.close(fig)
    return contents

def graphs_to_image(*args, title="", moving_average=True):
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.set_title(title)
    ax1.legend(loc='upper left')

    colors = ["r", "b"]

    if moving_average:
        for i, arg in enumerate(args):
            if len(arg) == 0:
                continue
            ax1.plot(arg, colors[i] + ".", alpha=0.2)

        for i, arg in enumerate(args):
            if len(arg) == 0:
                continue
            ax1.plot(util.moving_average(arg), colors[i])
    else:
        for i, arg in enumerate(args):
            if len(arg) == 0:
                continue
            ax1.plot(arg, colors[i])

    return figure_to_image(fig)

class TestServer(unittest.TestCase):

    def test_basic(self):
        FLAGS.image_server_storage_per_page = 100

        for i in range(10):
            data1 = np.random.uniform(low=0, high=1.0, size=(256, 256))
            data2 = np.random.uniform(low=0, high=1.0, size=(256, 256))
            data3 = np.random.uniform(low=0, high=1.0, size=(256, 256))
            put_images("delta", (data1, data2, data3))

        for i in range(100):
            data1 = np.random.uniform(low=0, high=1.0, size=(256, 256))
            data2 = np.random.uniform(low=0, high=1.0, size=(256, 256))
            data3 = np.random.uniform(low=0, high=1.0, size=(256, 256))
            put_images(
                ["alpha", "beta", "gamma"][i % 3], (data1, data2, data3))

        put_images("graph", [graphs_to_image("first",
                                             np.arange(100),
                                             np.sin(np.arange(100)),
                                             label="a")])

        put_images("graph", [graphs_to_image("second",
                                             np.arange(100),
                                             np.sqrt(np.arange(100)),
                                             label="b")])

        put_images("graph", [graphs_to_image("third",
                                             np.arange(100),
                                             np.sin(np.arange(100)),
                                             np.arange(100),
                                             np.sqrt(np.arange(100)),
                                             label=("c", "d"))])

        start()
        time.sleep(5)
        stop()

if __name__ == '__main__':
    FLAGS(sys.argv)
    util.setup_logging()
    unittest.main()
