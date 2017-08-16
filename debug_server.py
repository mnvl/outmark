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
gflags.DEFINE_integer("debug_server_max_rows", 20, "")

FLAGS = gflags.FLAGS

_rows = []
_lock = Lock()
_thread = None

class RequestHandler(BaseHTTPRequestHandler):
    def handle_index(self):
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()

        global _lock
        with _lock:
            text = "<center><table>"
            for i, (title, images) in enumerate(_rows):
                text += "<tr>"
                text += ("<td>%s</td>" % (title))
                for j in range(len(images)):
                    text += ("<td><img src=\"/%d/%d\"></td>" % (i, j))
                text += "</tr>"
            text += "</table></center>"

        self.wfile.write(text.encode("utf-8"))

    def handle_image(self):
        self.send_response(200)
        self.send_header('Content-type','image/png')
        self.end_headers()

        _, row, col = self.path.split("/")
        row = int(row)
        col = int(col)

        global _lock
        with _lock:
            data = _rows[row][1][col]
            misc.toimage(data).save(self.wfile, format = "png")

    def do_GET(self):
        if self.path.strip() == "/":
            self.handle_index()
            return

        self.handle_image()

        return

def start():
    server = HTTPServer(("", FLAGS.debug_server_port),
                        RequestHandler)
    global _thread
    _thread = Thread(target = server.serve_forever)
    _thread.start()

def put_images(title, images):
    global _lock
    with _lock:
        global _rows
        _rows.append((title, images))
        if len(_rows) > FLAGS.debug_server_max_rows:
            _rows = _rows[1:]

class TestServer(unittest.TestCase):

    def test_basic(self):
        for i in range(10):
            data1 = np.random.uniform(low = 0, high = 1.0, size = (256,256))
            data2 = np.random.uniform(low = 0, high = 1.0, size = (256,256))
            data3 = np.random.uniform(low = 0, high = 1.0, size = (256,256))
            put_images(str(i), (data1, data2, data3))

        start()

if __name__ == '__main__':
    FLAGS(sys.argv)
    util.setup_logging()
    unittest.main()
