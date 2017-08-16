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
gflags.DEFINE_integer("debug_server_max_rows", 5, "")

FLAGS = gflags.FLAGS

_table = {}
_lock = Lock()
_thread = None

class RequestHandler(BaseHTTPRequestHandler):
    def handle_index(self):
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()

        global _lock
        with _lock:
            text = "<meta http-equiv=\"refresh\" content=\"5\" />"
            text += "<center><table>"
            for title in sorted(_table.keys()):
                text += "<tr><td><b>%s</b></td></tr>" % title

                for i, images in enumerate(_table[title]):
                    text += "<tr>"
                    for j in range(len(images)):
                        text += ("<td><img src=\"/%s/%d/%d\"></td>" % (title, i, j))
                    text += "</tr>"
            text += "</table></center>"

        self.wfile.write(text.encode("utf-8"))

    def handle_image(self):
        self.send_response(200)
        self.send_header('Content-type','image/png')
        self.end_headers()

        _, title, row, col = self.path.split("/")
        row = int(row)
        col = int(col)

        global _lock
        with _lock:
            data = _table[title][row][col]
            misc.toimage(data).save(self.wfile, format = "png")

    def do_GET(self):
        if self.path.strip() == "/favicon.ico":
            self.send_response(404)
            return

        if self.path.strip() == "/":
            self.handle_index()
            return

        self.handle_image()

        return

    def log_message(self, format, *args):
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
        global _table
        _table[title] = _table.get(title, [])[-FLAGS.debug_server_max_rows:] + [images]

class TestServer(unittest.TestCase):

    def test_basic(self):
        for i in range(10):
            data1 = np.random.uniform(low = 0, high = 1.0, size = (256,256))
            data2 = np.random.uniform(low = 0, high = 1.0, size = (256,256))
            data3 = np.random.uniform(low = 0, high = 1.0, size = (256,256))
            put_images(str(i//5), (data1, data2, data3))

        start()

if __name__ == '__main__':
    FLAGS(sys.argv)
    util.setup_logging()
    unittest.main()
