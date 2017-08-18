#! /usr/bin/python3

import io
import sys
import time
import gflags
import logging
import unittest
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from scipy import misc
from threading import Thread, Lock
import hashlib
import util

gflags.DEFINE_integer("debug_server_port", 17171, "")
gflags.DEFINE_integer("debug_server_max_rows", 5, "")
gflags.DEFINE_integer("debug_server_max_images", 1000, "")

FLAGS = gflags.FLAGS

_server = None
_key_generator = 1
_table = {}
_images = {}
_queue = []
_lock = Lock()
_thread = None


class RequestHandler(BaseHTTPRequestHandler):

    def handle_index(self, page = None):
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
                p2 = "<font face=\"arial\" size=\"%d\">%s</font>" % (6 if p == page else 4, p)
                p2 = ("<a href=\"/index/%s\">%s</a>" % (p, p2)) if p != page else p2
                text += "<td>%s</td>" % p2
            text += "</tr></table></center>"

            text += "<center><table>"
            for images in _table[page]:
                text += "<tr>"
                for image in images:
                    text += (
                        "<td><img src=\"/image/%d\"></td>" % (image,))
                text += "</tr>"
            text += "</table></center>"

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
        logging.error("%s %s %s" % (self.address_string(), self.log_date_time_string(), fmt % args))

def start():
    global _server
    _server = HTTPServer(("", FLAGS.debug_server_port),
                        RequestHandler)
    global _thread
    _thread = Thread(target=_server.serve_forever)
    _thread.start()

def stop():
    global _server
    _server.shutdown()

def put_images(page, unencoded_images):
    global _key_generator
    global _lock
    global _queue

    images = []
    for image in unencoded_images:
        output = io.BytesIO()
        misc.toimage(image).save(output, format = "png")
        contents = output.getvalue()
        output.close()

        images.append(contents)

    with _lock:
        keys = []
        for i in range(len(images)):
            keys.append(_key_generator)
            _key_generator += 1

        _queue += keys

        while len(_queue) >= FLAGS.debug_server_max_images:
            _images[_queue[0]] = None
            _queue = _queue[1:]

        for k, i in zip(keys, images):
            _images[k] = i

        old = _table.get(page, [])[-FLAGS.debug_server_max_rows + 1:]
        _table[page] = old + [keys]


class TestServer(unittest.TestCase):

    def test_basic(self):
        for i in range(10):
            data1 = np.random.uniform(low=0, high=1.0, size=(256, 256))
            data2 = np.random.uniform(low=0, high=1.0, size=(256, 256))
            data3 = np.random.uniform(low=0, high=1.0, size=(256, 256))
            put_images(["alpha", "beta", "gamma"][i % 3], (data1, data2, data3))

        start()
        time.sleep(5)
        stop()

if __name__ == '__main__':
    FLAGS(sys.argv)
    util.setup_logging()
    unittest.main()
