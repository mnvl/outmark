
import logging
import unittest

import numpy as np
import tensorflow as tf
import gflags

class Model5:
  class Settings:
    D = 16
    H = 256
    W = 256
    C = 1

    batch_size = 10

    num_classes = 2

    num_conv_layers = 2
    num_conv_channels = 10

    num_dense_layers = 2
    num_dense_channels = 8

    kernel_size = 5

    learning_rate = 1e-4

    l2_reg = 1e-6

  def __init__(self, settings):
    self.S = settings

    self.session = tf.Session()

    self.X = tf.placeholder(tf.float32, shape=[self.S.batch_size, self.S.D, self.S.H, self.S.W, self.S.C])
    self.y = tf.placeholder(tf.uint8, shape=[self.S.batch_size, self.S.D, self.S.H, self.S.W, 1])
    logging.info("X: %s" % str(self.X))
    logging.info("y: %s" % str(self.y))

    self.conv_layers = []
    self.deconv_layers = []
    self.dense_layers = []

    Z = self.add_conv_layer("Input", self.X, False)
    self.conv_layers.append(Z)

    for i in range(0, self.S.num_conv_layers):
      Z = self.add_conv_layer("Conv%d" % i, Z)
      self.conv_layers.append(Z)

    for i in reversed(range(self.S.num_conv_layers)):
      Z = self.add_deconv_layer("Deconv%d" % i, Z)

      Z = tf.concat((Z, self.conv_layers[i]), 4)
      logging.info("Concat: %s ", str(Z))

      self.deconv_layers.append(Z)

    Z = self.conv_layers[0]

    for i in range(self.S.num_dense_layers):
      Z = self.add_dense_layer("Dense%d" % i, Z, i == self.S.num_dense_layers - 1)
      self.dense_layers.append(Z)

    DHW = self.S.D * self.S.H * self.S.W

    scores = tf.reshape(Z, [self.S.batch_size * DHW, -1])

    predictions_flat = tf.cast(tf.argmax(scores, axis = 1), tf.uint8)

    y_one_hot_flat = tf.one_hot(tf.reshape(self.y, [-1]), self.S.num_classes)

    self.loss = tf.nn.softmax_cross_entropy_with_logits(
      labels = y_one_hot_flat,
      logits = scores)
    self.loss = tf.reduce_mean(self.loss)

    self.train_step = tf.train.AdamOptimizer(
      learning_rate = self.S.learning_rate).minimize(self.loss)

    self.predictions = tf.reshape(predictions_flat, [self.S.batch_size, self.S.D, self.S.W, self.S.H])
    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.reshape(self.y, [-1]), predictions_flat), tf.float32))

    self.session.run(tf.global_variables_initializer())


    self.session.run(tf.global_variables_initializer())

  def add_conv_layer(self, name, Z, pool = True):
    with tf.variable_scope(name):
      W = tf.Variable(name = "W",
                      initial_value = tf.truncated_normal(
                        [3, self.S.kernel_size, self.S.kernel_size, int(Z.shape[4]), self.S.num_conv_channels],
                        dtype = tf.float32, stddev = 0.1))
      b = tf.Variable(name = "b",
                      initial_value = tf.constant(0.1, shape = (self.S.num_conv_channels,)))

      Z = tf.nn.conv3d(Z, W, [1, 1, 1, 1, 1], padding = "SAME") + b
      logging.info("%s: %s" % (name, str(Z)))

      Z = tf.nn.relu(Z)
      logging.info("%s: %s" % (name, str(Z)))

      if pool:
        Z = tf.nn.max_pool3d(Z, [1, 1, 1, 1, 1], [1, 2, 2, 2, 1], "SAME")
        logging.info("%s: %s" % (name, str(Z)))

      return Z

  def add_deconv_layer(self, name, Z):
    with tf.variable_scope(name):
      W = tf.Variable(name = "W",
                      initial_value = tf.truncated_normal(
                        [3, self.S.kernel_size, self.S.kernel_size, self.S.num_conv_channels, int(Z.shape[4])],
                        dtype = tf.float32, stddev = 0.1))
      b = tf.Variable(name = "b",
                      initial_value = tf.constant(0.1, shape = (self.S.num_conv_channels,)))

      Z = tf.nn.conv3d_transpose(Z, W,
                                 [self.S.batch_size, 2*int(Z.shape[1]), 2*int(Z.shape[2]), 2*int(Z.shape[3]), self.S.num_conv_channels],
                                 [1, 2, 2, 2, 1],
                                 padding = "SAME") + b
      logging.info("%s: %s" % (name, str(Z)))

      Z = tf.nn.relu(Z)
      logging.info("%s: %s" % (name, str(Z)))

      return Z

  def add_dense_layer(self, name, Z, last):
    output_channels = self.S.num_classes if last else self.S.num_dense_channels

    with tf.variable_scope(name):
      W = tf.Variable(name = "W",
                      initial_value = tf.truncated_normal(
                        [1, 1, 1, int(Z.shape[4]), output_channels],
                        dtype = tf.float32, stddev = 0.1))
      b = tf.Variable(name = "b",
                      initial_value = tf.constant(.0, shape = [output_channels]))

      Z = tf.nn.conv3d(Z, W, [1, 1, 1, 1, 1], "SAME") + b
      logging.info("%s: %s" % (name, str(Z)))

      if not last:
        Z = tf.nn.relu(Z)
        logging.info("%s: %s" % (name, str(Z)))

      return Z

  def fit(self, X, y):
    y = np.expand_dims(y, 4)

    (train_step, loss, accuracy) = self.session.run(
      [self.train_step, self.loss, self.accuracy],
      feed_dict = { self.X: X, self.y: y})
    return (loss, accuracy)

class TestModel5(unittest.TestCase):
  def test_overfit(self):
    D = 4

    settings = Model5.Settings()
    settings.num_classes = 2
    settings.batch_size = 1
    settings.H = settings.D = settings.W = D
    settings.C = 1
    settings.l2_reg = 0.
    settings.learning_rate = 0.01

    model = Model5(settings)

    X = np.random.randn(1, D, D, D, 1)
    y = (np.random.randn(1, D, D, D) > 0.5).astype(np.uint8)

    X[:, :, :, :, 0] -= .5 * y

    for i in range(100):
      loss, accuracy = model.fit(X, y)
      logging.info("step %d: loss = %f, accuracy = %f" % (i, loss, accuracy))

  def test_two(self):
    D = 32

    settings = Model5.Settings()
    settings.num_classes = 2
    settings.batch_size = 10
    settings.H = settings.D = settings.W = D
    settings.C = 1
    settings.num_conv_channels = 30
    settings.num_dense_channels = 30
    settings.l2_reg = 0.
    settings.learning_rate = 0.1

    model = Model5(settings)

    X = np.random.randn(settings.batch_size, D, D, D, 1)
    y = (np.random.randn(settings.batch_size, D, D, D) > 0.5).astype(np.uint8)

    X[:, :, :, :, 0] *= y

    for i in range(200):
      loss, accuracy = model.fit(X, y)
      logging.info("step %d: loss = %f, accuracy = %f" % (i, loss, accuracy))


if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')
  unittest.main()

