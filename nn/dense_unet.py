
import logging
import unittest
import numpy as np
import tensorflow as tf

class DenseUNet:
  class Settings:
    batch_size = 5
    num_classes = 2

    image_depth = 16
    image_height = 32
    image_width = 32
    image_channels = 1

    composite_kernel_size = 3
    composite_layers_in_block = 3
    growth_rate = 6
    reduction = 1.0

    init_features = growth_rate * 2
    init_kernel_size = 3

    bottleneck_features = growth_rate * 4

    num_blocks = 3

    dense_layers = 2

  def __init__(self, settings = Settings()):
    self.S = settings
    self.session = tf.Session()

  def define_placeholders(self):
    image_shape = [self.S.image_depth, self.S.image_height, self.S.image_width]
    self.X = tf.placeholder(tf.float32, shape=[self.S.batch_size] + image_shape + [self.S.image_channels])
    self.y = tf.placeholder(tf.uint8, shape=[self.S.batch_size] + image_shape + [1])
    logging.info("X: %s" % str(self.X))
    logging.info("y: %s" % str(self.y))

    self.is_training = tf.placeholder(tf.bool)

  def weight_variable(self, shape, name):
    return tf.get_variable(
      name=name,
      shape=shape,
      initializer=tf.contrib.layers.variance_scaling_initializer())

  def bias_variable(self, shape, name):
    return tf.get_variable(name, initializer=tf.constant(0.0, shape=shape))

  def batch_norm(self, inputs):
    return tf.layers.batch_normalization(inputs, training = self.is_training)

  def conv3d(self, inputs, output_channels, kernel_size, strides = [1, 1, 1, 1, 1]):
    input_channels = int(inputs.get_shape()[-1])
    W = self.weight_variable([kernel_size, kernel_size, kernel_size, input_channels, output_channels], "W")
    outputs = tf.nn.conv3d(inputs, W, strides, padding = "SAME")
    return outputs

  def deconv3d(self, inputs, output_shape, kernel_size, strides = [1, 2, 2, 2, 1]):
    input_channels = int(inputs.get_shape()[-1])
    W = self.weight_variable([kernel_size, kernel_size, kernel_size, input_channels, input_channels], "W")
    return tf.nn.conv3d_transpose(inputs, W, output_shape, strides, padding = "SAME")

  def avg_pool(self, inputs, k = 2):
    kernel_size = [1, k, k, k, 1]
    strides = [1, k, k, k, 1]
    return tf.nn.avg_pool3d(inputs, kernel_size, strides, "SAME")

  def upconv(self, inputs, k = 2):
    output_shape = [
      int(inputs.shape[0]),
      int(inputs.shape[1]) * k,
      int(inputs.shape[2]) * k,
      int(inputs.shape[3]) * k,
      int(inputs.shape[4]),
    ]
    return self.deconv3d(inputs, output_shape, 1)

  def composite(self, inputs, output_channels, kernel_size = None):
    if kernel_size is None: kernel_size = self.S.composite_kernel_size

    with tf.variable_scope("composite"):
      outputs = self.batch_norm(inputs)
      outputs = tf.nn.relu(outputs)
      outputs = self.conv3d(outputs, output_channels, kernel_size)
    return outputs

  def internal(self, inputs):
    composite_outputs = self.composite(inputs, self.S.growth_rate)
    outputs = tf.concat((inputs, composite_outputs), axis = -1)
    return outputs

  def block(self, inputs):
    outputs = inputs
    for layer in range(self.S.composite_layers_in_block):
      with tf.variable_scope("internal_%d" % layer):
        outputs = self.internal(outputs)
        logging.info(str(outputs))
    return outputs

  def transition_down(self, inputs):
    input_channels = int(inputs.get_shape()[-1])
    output_channels = int(input_channels * self.S.reduction)
    with tf.variable_scope("transition_down"):
      outputs = self.composite(inputs, output_channels, kernel_size = 1)
      outputs = self.avg_pool(outputs)
    return outputs

  def transition_up(self, inputs):
    input_channels = int(inputs.get_shape()[-1])
    output_channels = int(input_channels * self.S.reduction)
    with tf.variable_scope("transition_up"):
      outputs = self.upconv(inputs)
      outputs = self.composite(outputs, output_channels, kernel_size = 1)
    return outputs

  def bottleneck(self, inputs, output_features):
    with tf.variable_scope("bottleneck"):
      outputs = self.batch_norm(inputs)
      outputs = tf.nn.relu(outputs)
      outputs = self.conv3d(inputs, output_features, kernel_size = 1)
    return outputs

  def dense(self, inputs, output_channels):
    return self.composite(inputs, output_channels, kernel_size = 1)

  def add_layers(self):
    self.define_placeholders()

    outputs = self.X

    with tf.variable_scope("init"):
      outputs = self.conv3d(outputs, self.S.init_features, self.S.init_kernel_size)
      logging.info(str(outputs))

    block_outputs = []
    for layer in range(self.S.num_blocks):
      with tf.variable_scope("down_block_%d" % layer):
        outputs = self.block(outputs)
        logging.info(str(outputs))
        block_outputs.append(outputs)

        if layer != self.S.num_blocks - 1:
          outputs = self.transition_down(outputs)
          logging.info(str(outputs))
          outputs = self.bottleneck(outputs, self.S.bottleneck_features)
          logging.info(str(outputs))

    for layer in reversed(range(0, self.S.num_blocks - 1)):
      with tf.variable_scope("up_block_%d" % layer):
        outputs = self.transition_up(outputs)
        logging.info(str(outputs))
        outputs = tf.concat((outputs, block_outputs[layer]), axis = -1)
        outputs = self.block(outputs)
        logging.info(str(outputs))

    for layer in range(self.S.dense_layers):
      output_channels = self.S.num_classes if layer == self.S.dense_layers - 1 else int(outputs.get_shape()[-1])
      with tf.variable_scope("dense_%d" % layer):
        outputs = self.dense(outputs, output_channels)
        logging.info(str(outputs))

    self.layers_output = outputs

  def add_softmax_loss(self):
    Z = self.layers_output
    DHW = self.S.image_depth * self.S.image_height * self.S.image_width

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

    self.predictions = tf.reshape(predictions_flat, [self.S.batch_size, self.S.image_depth, self.S.image_height, self.S.image_width])
    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(y_flat, predictions_flat), tf.float32))

  def start(self):
    self.session.run(tf.global_variables_initializer())

  def stop(self):
    self.session.close()
    tf.reset_default_graph()

  def fit(self, X, y):
    y = np.expand_dims(y, 4)
    feed_dict = {
      self.X: X,
      self.y: y,
      self.is_training: True,
    }
    (train_step, loss, accuracy) = self.session.run(
      [self.train_step, self.loss, self.accuracy], feed_dict)
    return (loss, accuracy)

  def predict(self, X):
    (predictions) = self.session.run(
      [self.predictions],
      feed_dict = { self.X: X, self.is_training: False })
    return predictions

class TestDenseUNet(unittest.TestCase):
  def test_overfit(self):
    D = 8

    settings = DenseUNet.Settings()
    settings.num_classes = 2
    settings.batch_size = 1
    settings.image_width = settings.image_height = settings.image_depth = D
    settings.image_channels = 1
    settings.learning_rate = 0.01

    model = DenseUNet(settings)
    model.add_layers()
    model.add_softmax_loss()
    model.start()

    X = np.random.randn(1, D, D, D, 1)
    y = (np.random.randn(1, D, D, D) > 0.5).astype(np.uint8)

    X[:, :, :, :, 0] -= .5 * y

    for i in range(20):
      loss, accuracy = model.fit(X, y)
      logging.info("step %d: loss = %f, accuracy = %f" % (i, loss, accuracy))

    model.stop()

  def test_two(self):
    D = 16

    settings = DenseUNet.Settings()
    settings.num_classes = 10
    settings.batch_size = 10
    settings.image_width = settings.image_height = settings.image_depth = D
    settings.image_channels = 1
    settings.learning_rate = 0.01

    model = DenseUNet(settings)
    model.add_layers()
    model.add_softmax_loss()
    model.start()

    X = np.random.randn(settings.batch_size, D, D, D, 1)
    y = np.random.randint(0, 9, (settings.batch_size, D, D, D))

    X[:, :, :, :, 0] += y * 2

    for i in range(200):
      loss, accuracy = model.fit(X, y)
      if i % 20 == 0: logging.info("step %d: loss = %f, accuracy = %f" % (i, loss, accuracy))

    model.stop()

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')
  unittest.main()


