#! /usr/bin/python3

import numpy as np
import logging
import unittest
import gflags
import tensorflow as tf

class CNN:
  def __init__(self, D, H, W, minibatch_size, num_classes, conv1_size, conv2_size, conv3_size, fc1_size, fc2_size):
    self.D = D
    self.H = H
    self.W = W
    self.minibatch_size = minibatch_size
    self.num_classes = num_classes

    self.session = tf.Session()

    self.X = tf.placeholder(tf.float32, shape=[minibatch_size, D, H, W, 1])
    self.y = tf.placeholder(tf.float32, shape=[minibatch_size, num_classes])
    logging.info("X = %s" % str(self.X.shape))

    self.conv_W1 = tf.Variable(tf.truncated_normal([3, 3, 3, 1, conv1_size], stddev = 0.1))
    self.conv_b1 = tf.Variable(tf.constant(0.1, shape = [conv1_size]))
    self.conv_H1 = tf.nn.relu(tf.nn.conv3d(self.X, self.conv_W1, [1, 1, 1, 1, 1], "SAME") + self.conv_b1)
    logging.info("conv_H1 = %s" % str(self.conv_H1.shape))

    self.conv_W2 = tf.Variable(tf.truncated_normal([3, 3, 3, conv1_size, conv2_size], stddev = 0.1))
    self.conv_b2 = tf.Variable(tf.constant(0.1, shape = [conv2_size]))
    self.conv_H2 = tf.nn.relu(tf.nn.conv3d(self.conv_H1, self.conv_W2, [1, 1, 1, 1, 1], "SAME") + self.conv_b2)
    logging.info("conv_H2 = %s" % str(self.conv_H2.shape))

    self.conv_W3 = tf.Variable(tf.truncated_normal([3, 3, 3, conv2_size, conv3_size], stddev = 0.1))
    self.conv_b3 = tf.Variable(tf.constant(0.1, shape = [conv3_size]))
    self.conv_H3 = tf.nn.relu(tf.nn.conv3d(self.conv_H2, self.conv_W3, [1, 1, 1, 1, 1], "SAME") + self.conv_b3)
    logging.info("conv_H3 = %s" % str(self.conv_H2.shape))

    self.flat = tf.reshape(self.conv_H3, [minibatch_size, D * H * W * conv3_size])
    logging.info("flat = %s" % str(self.flat.shape))

    self.fc_W1 = tf.Variable(tf.truncated_normal([D * H * W * conv3_size, fc1_size], stddev = 0.1))
    self.fc_b1 = tf.Variable(tf.constant(0.1, shape = [fc1_size]))
    self.fc_H1 = tf.nn.relu(tf.matmul(self.flat, self.fc_W1) + self.fc_b1)
    logging.info("fc_H1 = %s" % str(self.fc_H1.shape))

    self.fc_W2 = tf.Variable(tf.truncated_normal([fc1_size, fc2_size], stddev = 0.1))
    self.fc_b2 = tf.Variable(tf.constant(0.1, shape=[fc2_size]))
    self.fc_H2 = tf.nn.relu(tf.matmul(self.fc_H1, self.fc_W2) + self.fc_b2)
    logging.info("fc_H2 = %s" % str(self.fc_H2.shape))

    self.fc_W3 = tf.Variable(tf.truncated_normal([fc2_size, num_classes], stddev = 0.1))
    self.fc_b3 = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    self.fc_H3 = tf.matmul(self.fc_H2, self.fc_W3) + self.fc_b3
    logging.info("fc_H3 = %s" % str(self.fc_H3.shape))

    self.cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self.fc_H3))

    self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
    self.correct_prediction = tf.equal(tf.argmax(self.fc_H3, 1), tf.argmax(self.y, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    self.session.run(tf.global_variables_initializer())

  def fit(self, X, y, X_val = None, y_val = None):
    with self.session.as_default():
      self.train_step.run(feed_dict={self.X: X, self.y: y})

      train_acc = self.accuracy.eval(feed_dict={self.X: X, self.y: y})

      if X_val is not None: val_acc = self.accuracy.eval(feed_dict={self.X: X_val, self.y: y_val})
      else: val_acc = None
      return (train_acc, val_acc)

class TestCNN(unittest.TestCase):
  def test_overfit_random(self):
    cnn = CNN(9, 9, 9, 20, 10, 40, 35, 30, 25, 20)

    X = np.random.randn(20, 9, 9, 9, 1)
    y = np.vstack([np.eye(10),np.eye(10)])

    for i in range(100):
      (accuracy, val_accuracy) = cnn.fit(X, y)
      if i % 10 == 0: logging.info("step %d: accuracy = %f" % (i, accuracy))

  def test_mismatching_size(self):
    cnn = CNN(19, 9, 19, 20, 10, 40, 35, 30, 25, 20)

    X = np.random.randn(20, 19, 9, 19, 1)
    y = np.vstack([np.eye(10),np.eye(10)])

    for i in range(100):
      (accuracy, val_accuracy) = cnn.fit(X, y)
      if i % 10 == 0: logging.info("step %d: accuracy = %f" % (i, accuracy))

  def test_val_accuracy(self):
    cnn = CNN(9, 9, 9, 20, 10, 40, 35, 30, 25, 20)

    X = np.random.randn(20, 9, 9, 9, 1)
    y = np.vstack([np.eye(10),np.eye(10)])

    X_val = X + .2 * np.random.randn(20, 9, 9, 9, 1)
    y_val = np.vstack([np.eye(10),np.eye(10)])

    for i in range(100):
      (accuracy, val_accuracy) = cnn.fit(X, y, X_val, y_val)
      if i % 10 == 0: logging.info("step %d: accuracy = %f, val_accuracy = %f" % (i, accuracy, val_accuracy))

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')
  unittest.main()
