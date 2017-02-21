
import numpy as np
import logging
import unittest
import gflags
import tensorflow as tf

class CNN2:
  def __init__(self, settings = {}):
    self.D = settings.get("D", 16)
    self.H = settings.get("H", 256)
    self.W = settings.get("W", 256)
    self.DHW = self.D * self.H * self.W
    self.minibatch_size = settings.get("minibatch_size", 10)
    self.num_classes = settings.get("num_classes", 2)
    self.num_channels = settings.get("num_channels", 1)
    self.conv1_size = settings.get("conv1_size", 100)
    self.conv2_size = settings.get("conv2_size", 100)
    self.conv3_size = settings.get("conv3_size", 100)
    self.conv4_size = settings.get("conv4_size", 100)
    self.conv5_size = settings.get("conv5_size", 100)
    self.learning_rate = settings.get("learning_rate", 1e-4)

    self.session = tf.Session()

    self.X = tf.placeholder(tf.float32, shape=(self.minibatch_size, self.D, self.H, self.W, self.num_channels))
    self.y = tf.placeholder(tf.uint8, shape=(self.minibatch_size, self.D, self.H, self.W, self.num_classes))
    logging.info("X = %s" % str(self.X.shape))
    logging.info("y = %s" % str(self.y.shape))

    self.conv_W1 = tf.Variable(tf.truncated_normal((3, 3, 3, self.num_channels, self.conv1_size), stddev = 0.1))
    self.conv_b1 = tf.Variable(tf.constant(0.1, shape = (self.conv1_size,)))
    self.conv_H1 = tf.nn.relu(tf.nn.conv3d(self.X, self.conv_W1, (1, 1, 1, 1, 1), "SAME") + self.conv_b1)
    logging.info("conv_H1 = %s" % str(self.conv_H1.shape))

    self.conv_W2 = tf.Variable(tf.truncated_normal((3, 3, 3, self.conv1_size, self.conv2_size), stddev = 0.1))
    self.conv_b2 = tf.Variable(tf.constant(0.1, shape = (self.conv2_size,)))
    self.conv_H2 = tf.nn.relu(tf.nn.conv3d(self.conv_H1, self.conv_W2, (1, 1, 1, 1, 1), "SAME") + self.conv_b2)
    logging.info("conv_H2 = %s" % str(self.conv_H2.shape))

    self.conv_W3 = tf.Variable(tf.truncated_normal((3, 3, 3, self.conv2_size, self.conv3_size), stddev = 0.1))
    self.conv_b3 = tf.Variable(tf.constant(0.1, shape = (self.conv3_size,)))
    self.conv_H3 = tf.nn.relu(tf.nn.conv3d(self.conv_H2, self.conv_W3, (1, 1, 1, 1, 1), "SAME") + self.conv_b3)
    logging.info("conv_H3 = %s" % str(self.conv_H3.shape))

    self.conv_W4 = tf.Variable(tf.truncated_normal((1, 1, 1, self.conv3_size, self.conv4_size), stddev = 0.1))
    self.conv_b4 = tf.Variable(tf.constant(0.1, shape = (self.conv4_size,)))
    self.conv_H4 = tf.nn.relu(tf.nn.conv3d(self.conv_H3, self.conv_W4, (1, 1, 1, 1, 1), "SAME") + self.conv_b4)
    logging.info("conv_H4 = %s" % str(self.conv_H3.shape))

    self.conv_W5 = tf.Variable(tf.truncated_normal((1, 1, 1, self.conv4_size, self.conv5_size), stddev = 0.1))
    self.conv_b5 = tf.Variable(tf.constant(0.1, shape = (self.conv5_size,)))
    self.conv_H5 = tf.nn.relu(tf.nn.conv3d(self.conv_H4, self.conv_W5, (1, 1, 1, 1, 1), "SAME") + self.conv_b5)
    logging.info("conv_H5 = %s" % str(self.conv_H3.shape))

    self.conv_W6 = tf.Variable(tf.truncated_normal((1, 1, 1, self.conv5_size, self.num_classes), stddev = 0.1))
    self.conv_b6 = tf.Variable(tf.constant(0.1, shape = (self.num_classes,)))
    self.conv_H6 = tf.nn.relu(tf.nn.conv3d(self.conv_H5, self.conv_W6, (1, 1, 1, 1, 1), "SAME") + self.conv_b6)
    logging.info("conv_H6 = %s" % str(self.conv_H6.shape))

    self.conv_last_flat = tf.reshape(self.conv_H6, (self.num_classes, self.minibatch_size * self.DHW))
    self.conv_last_flat = tf.transpose(self.conv_last_flat)
    logging.info("conv_last_flat = %s" % str(self.conv_last_flat.shape))

    self.y_flat = tf.reshape(self.y, (self.minibatch_size * self.DHW, self.num_classes))
    logging.info("y_flat = %s" % str(self.y_flat.shape))

    self.cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels = self.y_flat, logits = self.conv_last_flat))

    self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
    self.correct_prediction = tf.equal(tf.argmax(self.conv_last_flat, 1), tf.argmax(self.y_flat, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    self.session.run(tf.global_variables_initializer())

  def fit(self, X, y):
    with self.session.as_default():
      self.train_step.run(feed_dict={self.X: X, self.y: y})
      return self.accuracy.eval(feed_dict={self.X: X, self.y: y})

  def evaluate(self, X, y):
    with self.session.as_default():
      return self.accuracy.eval(feed_dict={self.X: X, self.y: y})

class TestCNN2(unittest.TestCase):
  def make_Xy(self, H, C):
    X = np.random.rand(1, 16, H, H, 1)

    N = 16*H*H
    y = np.zeros(shape = (N, C), dtype = np.uint8)
    y[np.arange(N), np.random.randint(0, C, N)] = 1
    y = y.reshape(1, 16, H, H, C)

    return (X, y)

  def test_overfit(self):
    H = 32
    cnn = CNN2({"num_classes": 10, "minibatch_size": 1, "H": H, "W": H})
    X, y = self.make_Xy(H, 10)
    for i in range(400):
      accuracy = cnn.fit(X, y)
      if i % 20 == 0: logging.info("step %d: accuracy = %f" % (i, accuracy))

  def test_run(self):
    H = 128
    cnn = CNN2({"num_classes": 10, "minibatch_size": 1, "H": H, "W": H})
    X, y = self.make_Xy(H, 10)
    accuracy = cnn.fit(X, y)
    logging.info("accuracy = %f" % (accuracy))

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')
  unittest.main()














