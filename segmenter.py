
import logging
import unittest
import numpy as np

def cut(get, put, z, y, x):
    for k in range(put.shape[0]):
        for j in range(put.shape[1]):
            for i in range(put.shape[2]):
                put[k, j, i] = get[min(z + k, get.shape[0] - 1),
                                   min(y + j, get.shape[1] - 1),
                                   min(x + i, get.shape[2] - 1)]

def paste(get, put, z, y, x):
    for k in range(get.shape[0]):
        if z + k >= put.shape[0]: continue

        for j in range(get.shape[1]):
            if y + j >= put.shape[1]: continue

            for i in range(get.shape[2]):
                if x + i >= put.shape[2]: continue

                put[z + k, y + j, x + i] = get[k, j, i]

def cut2(get, put, z, y, x):
    z = min(z, get.shape[0] - put.shape[0])
    y = min(y, get.shape[1] - put.shape[1])
    x = min(x, get.shape[2] - put.shape[2])

    put[:, :, :,] = get[z : z + put.shape[0],
                        y : y + put.shape[1],
                        x : x + put.shape[2]]

def paste2(get, put, z, y, x):
    z = min(z, put.shape[0] - get.shape[0])
    y = min(y, put.shape[1] - get.shape[1])
    x = min(x, put.shape[2] - get.shape[2])

    put[z : z + get.shape[0],
        y : y + get.shape[1],
        x : x + get.shape[2]] = get[:, :, :,]

class Segmenter:

    def __init__(self, predictor, input_depth, input_height, input_width, image):
        self.predictor = predictor
        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width
        self.image = image

    def predict(self):
        image_depth, image_height, image_width = self.image.shape

        prediction = np.zeros(self.image.shape, dtype = np.uint8)

        for z in range(0, image_depth, self.input_depth):
            for y in range(0, image_height, self.input_height):
                for x in range(0, image_width, self.input_width):
                    chunk = np.zeros((self.input_depth, self.input_height, self.input_width))
                    cut2(self.image, chunk, z, y, x)

                    chunk_prediction = self.predictor(chunk)

                    paste2(chunk_prediction, prediction, z, y, x)

        return prediction

class TestSegmenter(unittest.TestCase):

    def basic_test(self,
                   input_depth=1,
                   input_height=4,
                   input_width=3,
                   image_depth=1,
                   image_height=11,
                   image_width=11):
        def predictor(x):
            d, h, w = x.shape
            assert d == input_depth
            assert h == input_height
            assert w == input_width
            return x % 10

        image = np.random.randint(
            0, 200, (image_depth, image_height, image_width))
        label = image % 10

        segmenter = Segmenter(
            predictor, input_depth, input_height, input_width, image)
        prediction = segmenter.predict()

        assert np.all(prediction == label)

    def test_2d(self):
        self.basic_test()

    def test_3d(self):
        self.basic_test(image_depth=6)

    def test_big_2d(self):
        self.basic_test(input_height = 224,
                        input_width = 192,
                        image_height = 512,
                        image_width = 512)

    def test_big_3d(self):
        self.basic_test(input_depth = 8,
                        input_height = 224,
                        input_width = 192,
                        image_depth = 48,
                        image_height = 512,
                        image_width = 512)

    def test_strange(self):
        self.basic_test(input_depth = 4,
                        input_height = 8,
                        input_width = 6,
                        image_depth = 11,
                        image_height = 19,
                        image_width = 15)


if __name__ == '__main__':
    unittest.main()
