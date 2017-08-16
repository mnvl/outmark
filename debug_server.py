#! /usr/bin/python3

import sys
import gflags
import logging
import unittest
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from scipy import misc
from threading import Thread, Lock
import util

gflags.DEFINE_integer("debug_server_port", 17171, "")
gflags.DEFINE_integer("debug_server_max_images", 4, "")

FLAGS = gflags.FLAGS

_images = []
_lock = Lock()
_thread = None

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        logging.info("get request received")

        self.send_response(200)
        self.send_header('Content-type','image/png')
        self.end_headers()

        with _lock:
            rows = list(np.concatenate(x, axis = 0) for x in _images)
            data = np.concatenate(rows, axis = 1)

        print(data.shape)

        misc.toimage(data).save(self.wfile, format = "png")

        return

def start():
    server = HTTPServer(("", FLAGS.debug_server_port),
                        RequestHandler)
    global _thread
    _thread = Thread(target = server.serve_forever)
    _thread.start()

def put_images(images):
    global _lock
    with _lock:
        global _images
        _images.append(images)
        if len(_images) > FLAGS.debug_server_max_images:
            _images = _images[1:]

class TestServer(unittest.TestCase):

    def test_basic(self):
        for i in range(10):
            data1 = np.random.uniform(low = 0, high = 1.0, size = (256,256))
            data2 = np.random.uniform(low = 0, high = 1.0, size = (256,256))
            data3 = np.random.uniform(low = 0, high = 1.0, size = (256,256))
            put_images((data1, data2, data3))

        start()

if __name__ == '__main__':
    FLAGS(sys.argv)
    util.setup_logging()
    unittest.main()
