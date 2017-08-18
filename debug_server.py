#! /usr/bin/python3

import sys
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
gflags.DEFINE_integer("debug_server_max_images", 100, "")

FLAGS = gflags.FLAGS

_key_generator = 1
_table = {}
_images = {}
_queue = []
_lock = Lock()
_thread = None


class RequestHandler(BaseHTTPRequestHandler):

    def handle_index(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        global _lock
        with _lock:
            text = "<meta http-equiv=\"refresh\" content=\"5\" />"
            text += "<center><table>"
            for title in sorted(_table.keys()):
                text += "<tr><td><b>%s</b></td></tr>" % title

                for images in _table[title]:
                    text += "<tr>"
                    for image in images:
                        text += (
                            "<td><img src=\"/%d\"></td>" % (image,))
                    text += "</tr>"
            text += "</table></center>"

        self.wfile.write(text.encode("utf-8"))

    def handle_image(self):
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.end_headers()

        key = int(self.path.strip("/ "))

        global _lock
        with _lock:
            misc.toimage(_images[key]).save(self.wfile, format="png")

    def do_GET(self):
        if self.path.strip() == "/favicon.ico":
            self.send_response(404)
            return

        if self.path.strip() == "/":
            self.handle_index()
            return

        self.handle_image()

    def log_message(self, fmt, *args):
        logging.info("%s %s %s" % (self.address_string(), self.log_date_time_string(), fmt % args))

def start():
    server = HTTPServer(("", FLAGS.debug_server_port),
                        RequestHandler)
    global _thread
    _thread = Thread(target=server.serve_forever)
    _thread.start()


def put_images(title, images):
    global _key_generator
    global _lock
    global _queue

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

        old = _table.get(title, [])[-FLAGS.debug_server_max_rows + 1:]
        _table[title] = old + [keys]
        print(_table[title])


class TestServer(unittest.TestCase):

    def test_basic(self):
        for i in range(10):
            data1 = np.random.uniform(low=0, high=1.0, size=(256, 256))
            data2 = np.random.uniform(low=0, high=1.0, size=(256, 256))
            data3 = np.random.uniform(low=0, high=1.0, size=(256, 256))
            put_images(str(i // 5), (data1, data2, data3))

        start()

if __name__ == '__main__':
    FLAGS(sys.argv)
    util.setup_logging()
    unittest.main()
