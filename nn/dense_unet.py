
import logging
import unittest
import numpy as np
import tensorflow as tf

class DenseUNet:
  class Settings:
    batch_size = 5

    image_depth = 16
    image_height = 256
    image_width = 256
    image_channels = 1

    composite_kernel_size = 3
    layers_per_block = 3
    growth_rate = 12
    reduction = 1.0

  def __init__(self, settings = Settings()):
    self.S = settings

  def define_placeholders(self):
    image_shape = [self.S.image_depth, self.S.image_height, self.S.image_width]
    self.X = tf.placeholder(tf.float32, shape=[self.S.batch_size] + image_shape + [self.S.image_channels])
    self.y = tf.placeholder(tf.uint8, shape=[self.S.batch_size] + image_shape + [1])
    logging.info("X: %s" % str(self.X))
    logging.info("y: %s" % str(self.y))

    self.is_training = tf.placeholder(tf.bool)
    self.keep_prob = tf.placeholder(tf.float32)

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
    W = self.weight_variable([kernel_size, kernel_size, input_channels, output_channels])
    return tf.nn.conv3d(inputs, W, stride, padding = "SAME")

  def deconv3d(self, inputs, output_channels, kernel_size, strides = [1, 1, 1, 1, 1]):
    input_channels = int(inputs.get_shape()[-1])
    W = self.weight_variable([kernel_size, kernel_size, input_channels, output_channels])
    return tf.nn.conv3d_transpose(inputs, W, stride, padding = "SAME")

  def dropout(self, inputs):
    return tf.cond(
      self.is_training,
      lambda: tf.nn.dropout(inputs, self.keep_prob),
      inputs)

  def avg_pool(self, inputs, k = 2):
    kernel_size = [1, k, k, k, 1]
    strides = [1, k, k, k, 1]
    return tf.nn.avg_pool(inputs, kernel_size, strides, "SAME")

  def composite(self, inputs, output_channels, kernel_size = None):
    if kernel_size is None: kernel_size = self.S.composite_kernel_size

    with tf.variable_scope("composite"):
      outputs = self.batch_norm(inputs)
      outputs = tf.nn.relu(outputs)
      outputs = self.conv3d(outputs, output_channels, kernel_size)
      outputs = self.dropout(outputs)
    return outputs

  def bottleneck(self, inputs, output_channels):
    with tf.variable_scope("bottleneck"):
      outputs = self.batch_norm(inputs)
      outputs = tf.nn.relu(outputs)
      output = self.conv3d(
        outputs, out_features = output_channels * 4, kernel_size=1, padding="SAME")
      outputs = self.dropout(outputs)
    return outputs

  def transition(self, inputs):
    with tf.variable_scope("transition"):
      output_channels = int(int(inputs.get_shape()[-1]) * self.S.reduction)
      outputs = self.composite(inputs, output_channels, kernel_size = 1)
      outputs = self.avg_pool(outputs)
    return outputs

  def build_graph(self):
    self.define_placeholders()

class TestDenseUNet(unittest.TestCase):
  def test_build_graph(self):
    model = DenseUNet()
    model.build_graph()

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')
  unittest.main()


