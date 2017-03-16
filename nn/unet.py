
import logging
import unittest

import numpy as np
import tensorflow as tf
import gflags

# TODO: it should output one number for dice loss
class UNet:
  class Settings:
    image_depth = 16
    image_height = 128
    image_width = 128
    image_channels = 1

    batch_size = 10

    num_classes = 2
    class_weights = [1, 1]

    num_conv_blocks = 2
    num_conv_layers_per_block = 2
    num_conv_channels = 10

    num_dense_layers = 2
    num_dense_channels = 8

    kernel_size = 5

    learning_rate = 1e-4
    l2_reg = 1e-3

  def __init__(self, settings):
    self.S = settings

    self.session = tf.Session()

    self.X = tf.placeholder(tf.float32, shape=[self.S.batch_size, self.S.image_depth, self.S.image_height, self.S.image_width, self.S.image_channels])
    self.y = tf.placeholder(tf.uint8, shape=[self.S.batch_size, self.S.image_depth, self.S.image_height, self.S.image_width, 1])
    logging.info("X: %s" % str(self.X))
    logging.info("y: %s" % str(self.y))

    self.is_training = tf.placeholder(tf.bool)

    self.loss = 0

  def add_layers(self):
    self.conv_layers = []
    self.deconv_layers = []
    self.dense_layers = []

    with tf.variable_scope("init"):
      Z = self.add_conv_layer(self.X, output_channels = self.S.num_conv_channels)

    for i in range(0, self.S.num_conv_blocks):
      with tf.variable_scope("conv%d" % i):
        Z = self.add_conv_block(Z)
        self.conv_layers.append(Z)

        if i != self.S.num_conv_blocks - 1:
          Z = tf.nn.max_pool3d(Z, [1, 1, 1, 1, 1], [1, 1, 2, 2, 1], "SAME")
          logging.info("Pool: %s" % str(Z))

    for i in reversed(range(self.S.num_conv_blocks - 1)):
      with tf.variable_scope("deconv%d" % i):
        Z = self.add_deconv_block(Z)

        Z = tf.concat((Z, self.conv_layers[i]), 4)
        logging.info("Concat: %s ", str(Z))

        self.deconv_layers.append(Z)

    Z = self.batch_norm(Z)

    for i in range(self.S.num_dense_layers):
      Z = self.add_dense_layer("Dense%d" % i, Z, i == self.S.num_dense_layers - 1)
      self.dense_layers.append(Z)

  def add_softmax_loss(self):
    DHW = self.S.image_depth * self.S.image_height * self.S.image_width
    Z = self.dense_layers[-1]

    scores = tf.reshape(Z, [self.S.batch_size * DHW, self.S.num_classes])

    predictions_flat = tf.cast(tf.argmax(scores, axis = 1), tf.uint8)

    y_flat = tf.reshape(self.y, [-1])
    y_one_hot_flat = tf.one_hot(y_flat, self.S.num_classes)

    class_weights = tf.constant(np.array(self.S.class_weights, dtype=np.float32))
    logging.info("class_weights: %s" % str(class_weights))
    y_weights_flat = tf.reduce_sum(tf.multiply(class_weights, y_one_hot_flat), axis = 1)
    logging.info("y_weights_flat: %s" % str(y_weights_flat))

    softmax_loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_one_hot_flat, logits = scores)
    logging.info("softmax_loss: %s" % str(softmax_loss))

    self.loss += tf.reduce_mean(tf.multiply(softmax_loss, y_weights_flat))

    self.train_step = tf.train.AdamOptimizer(
      learning_rate = self.S.learning_rate).minimize(self.loss)

    self.predictions = tf.reshape(predictions_flat, [self.S.batch_size, self.S.image_depth, self.S.image_width, self.S.image_height])
    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(y_flat, predictions_flat), tf.float32))

  def start(self):
    self.session.run(tf.global_variables_initializer())

  def stop(self):
    self.session.close()
    tf.reset_default_graph()

  def weight_variable(self, shape, name):
    return tf.get_variable(
      name=name,
      shape=shape,
      initializer=tf.contrib.layers.variance_scaling_initializer())

  def bias_variable(self, shape, name):
    return tf.get_variable(name, initializer=tf.constant(0.0, shape=shape))

  def batch_norm(self, inputs):
    return tf.layers.batch_normalization(inputs, training = self.is_training)

  def add_conv_layer(self, Z, output_channels = None):
      if output_channels is None: output_channels = int(Z.shape[4])

      W = self.weight_variable([1, self.S.kernel_size, self.S.kernel_size, int(Z.shape[4]), output_channels], "W")
      b = self.bias_variable([output_channels], "b")

      self.loss += self.S.l2_reg * tf.reduce_sum(tf.square(W))

      Z = tf.nn.conv3d(Z, W, [1, 1, 1, 1, 1], padding = "SAME") + b
      logging.info(str(Z))

      Z = tf.nn.relu(Z)
      logging.info(str(Z))

      return Z

  def add_conv_block(self, Z):
    Z = self.batch_norm(Z)

    for layer in range(self.S.num_conv_layers_per_block):
      with tf.variable_scope("layer%d" % layer):
        Z = self.add_conv_layer(Z)
    return Z

  def add_deconv_layer(self, Z):
    output_channels = int(Z.shape[4])

    W = self.weight_variable([1, self.S.kernel_size, self.S.kernel_size, int(Z.shape[4]), output_channels], "W")
    b = self.bias_variable([output_channels], "b")

    self.loss += self.S.l2_reg * tf.reduce_sum(tf.square(W))

    Z = tf.nn.conv3d_transpose(Z, W,
                               [self.S.batch_size,
                                int(Z.shape[1]),
                                int(Z.shape[2]) * 2,
                                int(Z.shape[3]) * 2,
                                output_channels],
                               [1, 1, 2, 2, 1],
                               padding = "SAME") + b
    logging.info(str(Z))

    Z = tf.nn.relu(Z)
    logging.info(str(Z))

    return Z

  def add_deconv_block(self, Z):
    Z = self.batch_norm(Z)

    for layer in range(self.S.num_conv_layers_per_block):
      with tf.variable_scope("layer%d" % layer):
        if layer == 0: Z = self.add_deconv_layer(Z)
        else: Z = self.add_conv_layer(Z)
    return Z

  def add_dense_layer(self, name, Z, last):
    output_channels = self.S.num_classes if last else self.S.num_dense_channels

    with tf.variable_scope(name):
      W = self.weight_variable([1, 1, 1, int(Z.shape[4]), output_channels], "W")
      b = self.bias_variable([output_channels], "b")

      self.loss += self.S.l2_reg * tf.reduce_sum(tf.square(W))

      Z = tf.nn.conv3d(Z, W, [1, 1, 1, 1, 1], "SAME") + b
      logging.info("%s: %s" % (name, str(Z)))

      if not last:
        Z = tf.nn.relu(Z)
        logging.info("%s: %s" % (name, str(Z)))

      return Z

  def fit(self, X, y):
    y = np.expand_dims(y, 4)

    (_, loss, accuracy) = self.session.run(
      [self.train_step, self.loss, self.accuracy],
      feed_dict = { self.X: X, self.y: y, self.is_training: True})

    return (loss, accuracy)

  def predict(self, X):
    (predictions) = self.session.run(
      [self.predictions],
      feed_dict = { self.X: X, self.is_training: False })
    return predictions

  # X should be [depth, height, width, channels], depth may not be equal to self.S.image_depth
  def classify_image(self, image):
    image_depth = image.shape[0]
    depth_per_batch = self.S.image_depth * self.S.batch_size

    X = np.zeros([self.S.batch_size, self.S.image_depth, self.S.image_height, self.S.image_width, self.S.image_channels], dtype = np.float32)
    y = np.zeros([self.S.batch_size, self.S.image_depth, self.S.image_height, self.S.image_width], dtype = np.uint8)

    for i in range(0, 1 + image_depth // depth_per_batch):
      save = []

      for j in range(0, self.S.batch_size):
        base = i * depth_per_batch
        low = self.S.image_depth * j
        high = min(low + self.S.image_depth, image_depth - base)
        if high == low: break

        print(base, low, high)

        save.append((base, low, high))

        X[j, low : high, :, :, :] = image[low+base : high+base, :, :, :]
        if high - low < self.S.image_depth: X[j, base + high:, :, :, :] = X[j, base + high - 1, :, :, :]

      prediction = self.predict(X)

      for j, (base, low, high) in enumerate(save):
        print(base, low, high)
        y[low+base : high+base, :, :] = prediction[0][low:high, :, :]

class TestUNet(unittest.TestCase):
  def test_overfit(self):
    D = 4

    settings = UNet.Settings()
    settings.num_classes = 2
    settings.batch_size = 1
    settings.image_height = settings.image_depth = settings.image_width = D
    settings.image_channels = 1
    settings.learning_rate = 0.01

    model = UNet(settings)
    model.add_layers()
    model.add_softmax_loss()
    model.start()

    X = np.random.randn(1, D, D, D, 1)
    y = (np.random.randn(1, D, D, D) > 0.5).astype(np.uint8)

    X[:, :, :, :, 0] -= .5 * y

    for i in range(10):
      loss, accuracy = model.fit(X, y)
      logging.info("step %d: loss = %f, accuracy = %f" % (i, loss, accuracy))

    model.stop()

  def test_two(self):
    D = 16

    settings = UNet.Settings()
    settings.num_classes = 10
    settings.class_weights = [1] * 10
    settings.batch_size = 10
    settings.image_height = settings.image_depth = settings.image_width = D
    settings.image_channels = 1
    settings.num_conv_blocks = 3
    settings.num_conv_channels = 40
    settings.num_dense_channels = 40
    settings.learning_rate = 1e-3

    model = UNet(settings)
    model.add_layers()
    model.add_softmax_loss()
    model.start()

    X = np.random.randn(settings.batch_size, D, D, D, 1)
    y = np.random.randint(0, 9, (settings.batch_size, D, D, D))
    X[:, :, :, :, 0] += y * 2

    for i in range(100):
      loss, accuracy = model.fit(X, y)
      if i % 20 == 0: logging.info("step %d: loss = %f, accuracy = %f" % (i, loss, accuracy))

    model.stop()

  def test_overfit(self):
    D = 4

    settings = UNet.Settings()
    settings.num_classes = 2
    settings.batch_size = 1
    settings.image_height = settings.image_depth = settings.image_width = D
    settings.image_channels = 1
    settings.learning_rate = 0.01

    model = UNet(settings)
    model.add_layers()
    model.add_softmax_loss()
    model.start()

    X = np.random.randn(1, 7, D, D, 1)
    y = (np.random.randn(1, 7, D, D) > 0.5).astype(np.uint8)

    X[:, :, :, :, 0] *= np.sign(y)

    for i in range(10):
      loss, accuracy = model.fit(X[:, 0:D, :, :], y[:, 0:D, :, :])
      logging.info("step %d: loss = %f, accuracy = %f" % (i, loss, accuracy))

    #model.classify_image(X[0, :, :, :, :])

    model.stop()

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')
  unittest.main()

