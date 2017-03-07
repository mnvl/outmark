
import logging
import unittest

import numpy as np
import tensorflow as tf
import gflags

# TODO: it should output one number for dice loss
class ZNet:
  class Settings:
    D = 16
    H = 256
    W = 256
    C = 1

    batch_size = 10

    num_classes = 2

    num_conv_layers = 2
    num_conv_channels = 10

    extra_conv = True
    depth_max_pool = False

    num_dense_layers = 2
    num_dense_channels = 8

    kernel_size = 5

    learning_rate = 1e-4

  def __init__(self, settings):
    self.S = settings

    self.session = tf.Session()

    self.X = tf.placeholder(tf.float32, shape=[self.S.batch_size, self.S.D, self.S.H, self.S.W, self.S.C])
    self.y = tf.placeholder(tf.uint8, shape=[self.S.batch_size, self.S.D, self.S.H, self.S.W, 1])
    logging.info("X: %s" % str(self.X))
    logging.info("y: %s" % str(self.y))

  def add_layers(self):
    self.conv_layers = []
    self.deconv_layers = []
    self.dense_layers = []

    Z = self.X

    for i in range(0, self.S.num_conv_layers):
      Z = self.add_conv_layer("Conv%d" % i, Z)

      if self.S.extra_conv: Z = self.add_conv_layer("ExtraConv%d" % i, Z)

      self.conv_layers.append(Z)

      Z = tf.nn.max_pool3d(Z, [1, 1, 1, 1, 1], [1, 2 if self.S.depth_max_pool else 1, 2, 2, 1], "SAME")
      logging.info("Pool: %s" % str(Z))

    for i in reversed(range(self.S.num_conv_layers)):
      Z = self.add_deconv_layer("Deconv%d" % i, Z)

      if self.S.extra_conv: Z = self.add_conv_layer("ExtraDeconv%d" % i, Z)

      Z = tf.concat((Z, self.conv_layers[i]), 4)
      logging.info("Concat: %s ", str(Z))

      self.deconv_layers.append(Z)

    Z = self.conv_layers[0]

    for i in range(self.S.num_dense_layers):
      Z = self.add_dense_layer("Dense%d" % i, Z, i == self.S.num_dense_layers - 1)
      self.dense_layers.append(Z)

  def add_dice_loss(self):
    DHW = self.S.D * self.S.H * self.S.W
    Z = self.dense_layers[-1]

    scores = tf.reshape(tf.sigmoid(Z), [self.S.batch_size * DHW])
    y_flat = tf.reshape(self.y, [self.S.batch_size * DHW])

    y_flat2 = tf.cast(y_flat, tf.float32)
    self.loss = -(2. * tf.reduce_sum(scores * y_flat2) + 1.) / (tf.reduce_sum(scores) + tf.reduce_sum(y_flat2) + 1.)

    self.train_step = tf.train.AdamOptimizer(
      learning_rate = self.S.learning_rate).minimize(self.loss)

    predictions_flat = tf.cast(scores > tf.reduce_mean(scores) , tf.uint8)
    self.predictions = tf.reshape(scores, [self.S.batch_size, self.S.D, self.S.W, self.S.H])

    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(y_flat, predictions_flat), tf.float32))

  def add_softmax_loss(self):
    DHW = self.S.D * self.S.H * self.S.W
    Z = self.dense_layers[-1]

    scores = tf.reshape(Z, [self.S.batch_size * DHW, self.S.num_classes])

    predictions_flat = tf.cast(tf.argmax(scores, axis = 1), tf.uint8)

    y_flat = tf.reshape(self.y, [-1])
    y_one_hot_flat = tf.one_hot(y_flat, self.S.num_classes)

    self.loss = tf.nn.softmax_cross_entropy_with_logits(
      labels = y_one_hot_flat,
      logits = scores)
    self.loss = tf.reduce_mean(self.loss)

    self.train_step = tf.train.AdamOptimizer(
      learning_rate = self.S.learning_rate).minimize(self.loss)

    self.predictions = tf.reshape(predictions_flat, [self.S.batch_size, self.S.D, self.S.W, self.S.H])
    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(y_flat, predictions_flat), tf.float32))

  def init(self):
    self.session.run(tf.global_variables_initializer())

  def add_conv_layer(self, name, Z):
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
                                 [self.S.batch_size,
                                  (2 if self.S.depth_max_pool else 1)*int(Z.shape[1]),
                                  2*int(Z.shape[2]),
                                  2*int(Z.shape[3]),
                                  self.S.num_conv_channels],
                                 [1, 2 if self.S.depth_max_pool else 1, 2, 2, 1],
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

  def predict(self, X):
    (predictions) = self.session.run(
      [self.predictions],
      feed_dict = { self.X: X })
    return predictions

class TestZNet(unittest.TestCase):
  def test_overfit(self):
    D = 4

    settings = ZNet.Settings()
    settings.num_classes = 2
    settings.batch_size = 1
    settings.H = settings.D = settings.W = D
    settings.C = 1
    settings.learning_rate = 0.01

    model = ZNet(settings)
    model.add_layers()
    model.add_softmax_loss()
    model.init()

    X = np.random.randn(1, D, D, D, 1)
    y = (np.random.randn(1, D, D, D) > 0.5).astype(np.uint8)

    X[:, :, :, :, 0] -= .5 * y

    for i in range(20):
      loss, accuracy = model.fit(X, y)
      logging.info("step %d: loss = %f, accuracy = %f" % (i, loss, accuracy))

  def test_two(self):
    D = 16

    settings = ZNet.Settings()
    settings.num_classes = 10
    settings.batch_size = 10
    settings.H = settings.D = settings.W = D
    settings.C = 1
    settings.num_conv_layers = 2
    settings.num_conv_channels = 20
    settings.num_dense_channels = 20
    settings.learning_rate = 1e-4

    model = ZNet(settings)
    model.add_layers()
    model.add_softmax_loss()
    model.init()

    X = np.random.randn(settings.batch_size, D, D, D, 1)
    y = np.random.randint(0, 9, (settings.batch_size, D, D, D))

    X[:, :, :, :, 0] += y * 2

    for i in range(200):
      loss, accuracy = model.fit(X, y)

      if i % 20 == 0: logging.info("step %d: loss = %f, accuracy = %f" % (i, loss, accuracy))


if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')
  unittest.main()

