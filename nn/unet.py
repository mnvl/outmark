
import logging
import unittest
import numpy as np
import tensorflow as tf
import gflags
import util

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

    kernel_size = 3

    learning_rate = 1e-4
    l2_reg = 1e-3

    use_batch_norm = False
    keep_prob = 0.9

  def __init__(self, settings):
    self.S = settings

    self.session = tf.Session()

    self.X = tf.placeholder(tf.float32, shape=[self.S.batch_size, self.S.image_depth, self.S.image_height, self.S.image_width, self.S.image_channels])
    self.y = tf.placeholder(tf.uint8, shape=[self.S.batch_size, self.S.image_depth, self.S.image_height, self.S.image_width, 1])
    logging.info("X: %s" % str(self.X))
    logging.info("y: %s" % str(self.y))

    self.is_training = tf.placeholder(tf.bool)
    self.keep_prob = tf.placeholder(tf.float32)

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
          ksize = [1, 1, 2, 2, 1]
          strides = [1, 1, 2, 2, 1]
          Z = tf.nn.max_pool3d(Z, ksize, strides, "SAME")
          logging.info("Pool: %s" % str(Z))

    for i in reversed(range(self.S.num_conv_blocks - 1)):
      with tf.variable_scope("deconv%d" % i):
        Z = self.add_deconv_block(Z, self.conv_layers[i])
        self.deconv_layers.append(Z)

    Z = self.batch_norm(Z)
    logging.info(str(Z))

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

    y_flat_nonzero = tf.cast(tf.not_equal(y_flat, 0), tf.float32)
    predictions_flat_nonzero = tf.cast(tf.not_equal(predictions_flat, 0), tf.float32)
    dice_nominator = tf.cast(tf.equal(y_flat, predictions_flat), tf.float32)
    dice_nominator = tf.multiply(dice_nominator, y_flat_nonzero)
    dice_nominator = tf.multiply(dice_nominator, predictions_flat_nonzero)
    dice_nominator = tf.multiply(2., tf.reduce_sum(dice_nominator))
    dice_denominator = 1.0
    dice_denominator = tf.add(dice_denominator, tf.reduce_sum(y_flat_nonzero))
    dice_denominator = tf.add(dice_denominator, tf.reduce_sum(predictions_flat_nonzero))
    self.dice = tf.divide(dice_nominator, dice_denominator)

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
    if self.S.use_batch_norm:
      return tf.layers.batch_normalization(inputs, training = self.is_training)
    else:
      return inputs

  def dropout(self, inputs):
    return tf.cond(
      self.is_training,
      lambda: tf.nn.dropout(inputs, self.keep_prob),
      lambda: inputs)

  def add_conv_layer(self, Z, output_channels = None):
      if output_channels is None: output_channels = int(Z.shape[4])

      W = self.weight_variable([1, self.S.kernel_size, self.S.kernel_size, int(Z.shape[4]), output_channels], "W")
      b = self.bias_variable([output_channels], "b")

      self.loss += tf.multiply(self.S.l2_reg, tf.reduce_sum(tf.square(W)))

      Z = tf.nn.conv3d(Z, W, [1, 1, 1, 1, 1], padding = "SAME") + b
      logging.info(str(Z))

      Z = tf.nn.relu(Z)
      logging.info(str(Z))

      Z = self.dropout(Z)

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

    self.loss += tf.multiply(self.S.l2_reg, tf.reduce_sum(tf.square(W)))

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

  def add_deconv_block(self, Z, cc):
    Z = self.batch_norm(Z)

    Z = self.add_deconv_layer(Z)
    logging.info("Deconv: %s" % (str(Z)))

    Z = tf.concat((Z, cc), axis = 4)
    logging.info("Concat: %s" % (str(Z)))

    for layer in range(self.S.num_conv_layers_per_block):
      with tf.variable_scope("layer%d" % layer):
        Z = self.add_conv_layer(Z)
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

        Z = self.dropout(Z)
        logging.info("%s: %s" % (name, str(Z)))

      return Z

  def fit(self, X, y):
    y = np.expand_dims(y, 4)
    (_, loss, accuracy, dice) = self.session.run(
      [self.train_step, self.loss, self.accuracy, self.dice],
      feed_dict = { self.X: X, self.y: y, self.is_training: True, self.keep_prob: self.S.keep_prob})

    return (loss, accuracy, dice)

  def predict(self, X, y = None):
    if y is None:
      (predictions,) = self.session.run(
        [self.predictions],
        feed_dict = { self.X: X, self.is_training: False, self.keep_prob: self.S.keep_prob })
      return predictions
    else:
      y = np.expand_dims(y, 4)
      (predictions, loss, accuracy, dice) = self.session.run(
        [self.predictions, self.loss, self.accuracy, self.dice],
        feed_dict = { self.X: X, self.y: y, self.is_training: False, self.keep_prob: self.S.keep_prob })
      return (predictions, loss, accuracy, dice)

  # X should be [depth, height, width, channels], depth may not be equal to self.S.image_depth
  def segment_image(self, image):
    image_depth = image.shape[0]
    depth_per_batch = self.S.image_depth * self.S.batch_size

    X = np.zeros([self.S.batch_size, self.S.image_depth, self.S.image_height, self.S.image_width, self.S.image_channels], dtype = np.float32)
    result = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype = np.uint8)

    for i in range(0, 1 + image_depth // depth_per_batch):
      save = []

      for j in range(0, self.S.batch_size):
        low = i * depth_per_batch + self.S.image_depth * j
        high = min(low + self.S.image_depth, image_depth)
        if low >= high: break

        save.append((low, high))

        X[j, : high-low, :, :, :] = image[low:high, :, :, :]

        if high - low < self.S.image_depth:
          for k in range(high - low, self.S.image_depth):
            X[j, k, :, :, :] = image[high - 1, :, :, :]

      prediction = self.predict(X)

      for j, (low, high) in enumerate(save):
        result[low:high, :, :] = prediction[j, :high-low, :, :]

    return result

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
      loss, accuracy, dice = model.fit(X, y)
      logging.info("step %d: loss = %f, accuracy = %f" % (i, loss, accuracy))

    model.stop()

  def test_two(self):
    D = 8

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
      loss, accuracy, dice = model.fit(X, y)
      if i % 20 == 0: logging.info("step %d: loss = %f, accuracy = %f" % (i, loss, accuracy))

    model.stop()

  def test_metrics_two_classes(self):
    D = 4

    settings = UNet.Settings()
    settings.num_classes = 2
    settings.class_weights = [1] * 2
    settings.batch_size = 10
    settings.image_height = settings.image_depth = settings.image_width = D
    settings.image_channels = 1
    settings.num_conv_blocks = 3
    settings.num_conv_channels = 40
    settings.num_dense_channels = 40
    settings.learning_rate = 0

    model = UNet(settings)
    model.add_layers()
    model.add_softmax_loss()
    model.start()

    for i in range(10):
      X = np.random.randn(settings.batch_size, D, D, D, 1)
      y = np.random.randint(0, 2, (settings.batch_size, D, D, D), dtype = np.uint)

      y_pred, loss, accuracy, dice = model.predict(X, y)

      accuracy2 = util.accuracy(y_pred, y)
      dice2 = util.dice(y_pred, y)
      
      logging.info("batch %d: loss = %f, accuracy = %f, accuracy2 = %f, dice = %f, dice2 = %f" % (i, loss, accuracy, accuracy2, dice, dice2))

      assert abs(accuracy - accuracy2) < 0.001, "accuracy mismatch!"
      assert abs(dice - dice2) < 0.001, "dice mismatch!"

    model.stop()

  def test_metrics_many_classes(self):
    D = 4

    settings = UNet.Settings()
    settings.num_classes = 10
    settings.class_weights = [1] * 10
    settings.batch_size = 10
    settings.image_height = settings.image_depth = settings.image_width = D
    settings.image_channels = 1
    settings.num_conv_blocks = 3
    settings.num_conv_channels = 40
    settings.num_dense_channels = 40
    settings.learning_rate = 0

    model = UNet(settings)
    model.add_layers()
    model.add_softmax_loss()
    model.start()

    for i in range(10):
      X = np.random.randn(settings.batch_size, D, D, D, 1)
      y = np.random.randint(0, 10, (settings.batch_size, D, D, D), dtype = np.uint)

      y_pred, loss, accuracy, dice = model.predict(X, y)

      accuracy2 = util.accuracy(y_pred, y)
      dice2 = util.dice(y_pred, y)
      
      logging.info("batch %d: loss = %f, accuracy = %f, accuracy2 = %f, dice = %f, dice2 = %f" % (i, loss, accuracy, accuracy2, dice, dice2))

      assert abs(accuracy - accuracy2) < 0.001, "accuracy mismatch!"
      assert abs(dice - dice2) < 0.001, "dice mismatch!"

    model.stop()

  def test_segment_image(self):
    D = 4
    B = 15

    settings = UNet.Settings()
    settings.num_classes = 2
    settings.class_weights = [1., 1.]
    settings.image_height = settings.image_width = D
    settings.image_depth = 1
    settings.batch_size = B 
    settings.image_channels = 1
    settings.learning_rate = 0.01
    settings.l2_reg = 0.1

    model = UNet(settings)
    model.add_layers()
    model.add_softmax_loss()
    model.start()

    X_val = np.random.randn(B, D, D, 1)
    y_val = (np.random.randn(B, D, D) > 0.5).astype(np.uint8)
    X_val[:, :, :, 0] += 10. * y_val

    for i in range(200):
      X = np.random.randn(settings.batch_size, 1, D, D, 1)
      y = (np.random.randn(settings.batch_size, 1, D, D) > 0.5).astype(np.uint8)
      X[:, :, :, :, 0] += 10. * y

      y_pred = model.predict(X)

      val_accuracy = util.accuracy(y_pred, y)
      val_dice = util.dice(y_pred, y)

      y_seg = model.segment_image(np.squeeze(X, axis = 1))
      assert((y_seg == np.squeeze(y_pred, axis = 1)).all()), \
        "Segmenting error: " + str(y_seg != y_pred)

      loss, accuracy, dice = model.fit(X, y)

      if (i + 1) % 10 == 0 or i == 0:
        logging.info("step %d: loss = %f, accuracy = %f, dice = %f, "
                     "val_accuracy = %f, val_dice = %f" %
                     (i, loss, accuracy, dice, val_accuracy, val_dice))

    seg_pred = model.segment_image(X_val)
    seg_acc = util.accuracy(seg_pred, y_val)
    seg_dice = util.dice(seg_pred, y_val)
    logging.info("segmentation accuracy = %f, dice = %f", seg_acc, seg_dice)

    assert abs(seg_acc - val_accuracy) < 0.1,\
      "something is wrong here! segmentation code might be broken, or it's just flaky test"

    assert abs(seg_dice - val_dice) < 0.1,\
      "something is wrong here! segmentation code might be broken, or it's just flaky test"

    model.stop()

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(levelname)s %(message)s',
                      filename='/dev/stderr',
                      filemode='w')
  unittest.main()

