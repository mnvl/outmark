
import logging
import unittest
import numpy as np
import tensorflow as tf
import util

EPSILON = 1.0e-6


def accuracy(a, b):
    return np.mean((a.flatten() == b.flatten()).astype(np.float32))


def iou(a, b, num_classes):
    assert a.shape == b.shape

    if len(a.shape) == 1:
        a = np.expand_dims(a, 0)
        b = np.expand_dims(b, 0)

    if len(a.shape) > 3:
        a = a.reshape((a.shape[0], -1, a.shape[-1]))
        b = b.reshape((b.shape[0], -1, b.shape[-1]))

    s = np.zeros(a.shape[0])
    for c in range(1, num_classes):
        i = np.sum(np.logical_and(a == c, b == c), dtype=np.float32)
        u = np.sum(np.logical_or(a == c, b == c), dtype=np.float32)
        s += i / (u + EPSILON)

    return np.mean(s / (num_classes - 1.0))


def iou_op(x, y):
    # http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
    assert len(x.shape) == 3
    assert len(y.shape) == 3

    i = tf.reduce_sum(x * y, axis=1)
    u = tf.reduce_sum(x, axis=1) + tf.reduce_sum(y, axis=1) - i

    return tf.reduce_mean(i / (u + EPSILON), axis=0)


class TestIOU(unittest.TestCase):

    def setUp(self):
        self.session = tf.Session()
        self.x = tf.placeholder(tf.float32, shape=[None, None, None])
        self.y = tf.placeholder(tf.float32, shape=[None, None, None])
        self.iou = iou_op(self.x, self.y)
        self.mean_iou = tf.reduce_mean(self.iou)
        self.mean_iou_grad = tf.gradients(self.mean_iou, [self.x, self.y])

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def calculate_iou(self, x, y):
        x = np.array(x)
        y = np.array(y)
        iou, mean_iou, mean_iou_grad = self.session.run(
            [self.iou, self.mean_iou, self.mean_iou_grad], {self.x: x, self.y: y})
        return iou, mean_iou, mean_iou_grad

    def test_correctness(self):
        x = np.array([[[1.0, 0.0, 0.0],
                       [0.0, 1.0, 1.0]]])
        y = np.array([[[1.0, 1.0, 0.0],
                       [0.0, 1.0, 1.0]]])
        iou1, _, _ = self.calculate_iou(x, y)
        assert np.allclose(iou1, [1.0, 0.5, 1.0]), iou

    def test_correctness_multiple_examples(self):
        iou, _, _ = self.calculate_iou(
            [[[1.0, 0.0, 0.0],
              [0.0, 1.0, 1.0]],
             [[1.0, 0.0, 0.0],
              [0.0, 1.0, 1.0]]],
            [[[1.0, 1.0, 0.0],
              [0.0, 1.0, 1.0]],
             [[0.5, 0.7, 0.9],
              [0.5, 0.7, 0.9]]])
        assert np.allclose(
            iou, [(1.0 + 0.5 / 1.5) / 2,
                  (0.5 + 0.7 / 1.7) / 2,
                  (1.0 + 0.9 / 1.9) / 2]), iou

    def test_gradients(self):
        shape = (2, 3, 5)

        def f(args):
            x = args[:np.prod(shape)].reshape(shape)
            y = args[np.prod(shape):].reshape(shape)
            iou, mean_iou, mean_iou_grad = self.calculate_iou(x, y)
            mean_iou_grad = np.array(mean_iou_grad).reshape(-1)
            return mean_iou, mean_iou_grad

        args = np.random.uniform(size=2 * np.prod(shape))
        delta = 1.0e-3
        for i in range(args.shape[0]):
            iou, analytic_grad = f(args)

            args[i] -= delta
            iou1, _ = f(args)

            args[i] += 2 * delta
            iou2, _ = f(args)

            analytic_grad = analytic_grad[i]
            numeric_grad = (iou2 - iou1) / (2 * delta)

            assert abs(analytic_grad - numeric_grad) < 0.001

    def test_correctness_cpu(self):
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 1, 0, 1])
        assert abs(iou(x, y, 2) - 1 / 3) < 0.001

if __name__ == '__main__':
    util.setup_logging()
    unittest.main()
