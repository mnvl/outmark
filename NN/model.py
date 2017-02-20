#! /usr/bin/python3

import numpy as np
import logging
import unittest
import gflags
import tensorflow as tf

class CNN:
  def __init__(self, D, H, W, minibatch_size, num_classes, conv1_size, conv2_size, fc1_size, fc2_size):
    self.D = D
    self.H = H
    self.W = W
    self.minibatch_size = minibatch_size
    self.num_classes = num_classes

    self.session = tf.Session()

    self.X = tf.placeholder(tf.float32, shape=[minibatch_size, D, H, W, 1])
    self.y = tf.placeholder(tf.float32, shape=[minibatch_size, num_classes])

    self.W1 = tf.Variable(tf.truncated_normal([3, 3, 3, 1, conv1_size], stddev = 0.1))
    self.b1 = tf.Variable(tf.constant(0.1, shape = [conv1_size]))
    self.H1 = tf.nn.relu(tf.nn.conv3d(self.X, self.W1, [1, 1, 1, 1, 1], "SAME") + self.b1)

    self.W2 = tf.Variable(tf.truncated_normal([3, 3, 3, conv1_size, conv2_size], stddev = 0.1))
    self.b2 = tf.Variable(tf.constant(0.1, shape = [conv2_size]))
    self.H2 = tf.nn.relu(tf.nn.conv3d(self.H1, self.W2, [1, 1, 1, 1, 1], "SAME") + self.b2)

    self.flat = tf.reshape(self.H2, [minibatch_size, D * H * W * conv2_size])

    self.W3 = tf.Variable(tf.truncated_normal([D * H * W * conv2_size, fc1_size], stddev = 0.1))
    self.b3 = tf.Variable(tf.constant(0.1, shape = [fc1_size]))
    self.H3 = tf.nn.relu(tf.matmul(self.flat, self.W3) + self.b3)

    self.W4 = tf.Variable(tf.truncated_normal([fc1_size, fc2_size], stddev = 0.1))
    self.b4 = tf.Variable(tf.constant(0.1, shape=[fc2_size]))
    self.H4 = tf.nn.relu(tf.matmul(self.H3, self.W4) + self.b4)

    self.W5 = tf.Variable(tf.truncated_normal([fc2_size, num_classes], stddev = 0.1))
    self.b5 = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    self.H5 = tf.matmul(self.H4, self.W5) + self.b5

    self.cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self.H5))

    self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
    self.correct_prediction = tf.equal(tf.argmax(self.H5, 1), tf.argmax(self.y, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    self.session.run(tf.global_variables_initializer())

  def fit(self, X, y):
    with self.session.as_default():
      self.train_step.run(feed_dict={self.X: X, self.y: y})
      accuracy = self.accuracy.eval(feed_dict={self.X: X, self.y: y})
      return accuracy

class TestCNN(unittest.TestCase):
  def test_overfit_random(self):
    cnn = CNN(9, 9, 9, 20, 10, 40, 35, 30, 25)

    X = np.random.randn(20, 9, 9, 9, 1)
    y = np.vstack([np.eye(10),np.eye(10)])

    for i in range(100):
      accuracy = cnn.fit(X, y)
      if i % 10 == 0: logging.info("step %d: accuracy = %f" % (i, accuracy))

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')
  unittest.main()
