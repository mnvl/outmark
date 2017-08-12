
import sys
import logging
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function as tf_function
import tflearn
import gflags
from timeit import default_timer as timer
import util
from segmenter import Segmenter


gflags.DEFINE_string("summary", "./summary/", "")

FLAGS = gflags.FLAGS


# http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
def iou_op_grad(op, grad):
    x, y = op.inputs

    i = tf.reduce_sum(x * y, axis=0)
    u = tf.reduce_sum(x, axis=0) + tf.reduce_sum(y, axis=0) - i

    k1 = 1.0 / (u + 1.0)
    k2 = -i / (tf.square(u) + 1.0)

    x_grad = k1 * y + k2 * (1.0 - y)
    y_grad = k1 * x + k2 * (1.0 - x)

    return x_grad * grad, y_grad * grad


@tf_function.Defun(tf.float32, tf.float32, python_grad_func=iou_op_grad)
def iou_op(x, y):
    i = tf.reduce_sum(x[:, 1:] * y[:, 1:], axis=0)
    u = tf.reduce_sum(x[:, 1:], axis=0) + tf.reduce_sum(y[:, 1:], axis=0) - i

    return tf.reduce_mean(i / (u + 1.0))


class VNet:

    class Settings:
        image_depth = 1
        image_height = 224
        image_width = 224

        num_classes = 2
        class_weights = [1, 1]

        num_conv_blocks = 2
        num_conv_channels = 10

        num_dense_layers = 0
        num_dense_channels = 8

        learning_rate = 1e-4
        l2_reg = 1e-5

        loss = "softmax"

        use_batch_norm = False
        keep_prob = 0.9

    def __init__(self, settings):
        self.S = settings

        self.session = tf.Session(
            config=tf.ConfigProto(log_device_placement=False,))

        self.X = tf.placeholder(
            tf.float32, shape=[None, self.S.image_depth, self.S.image_height, self.S.image_width])
        self.y = tf.placeholder(
            tf.uint8, shape=[None, self.S.image_depth, self.S.image_height, self.S.image_width])
        logging.info("X: %s" % str(self.X))
        logging.info("y: %s" % str(self.y))

        self.is_training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)

    def add_layers(self):
        self.loss = 0
        self.conv_layers = []
        self.deconv_layers = []
        self.dense_layers = []

        Z = tf.expand_dims(self.X, 4)
        logging.info(str(Z))

        for i in range(0, self.S.num_conv_blocks):
            num_channels = self.S.num_conv_channels * (2 ** i)

            with tf.variable_scope("conv%d" % i):
                Z = self.add_conv_block(Z, channels=num_channels)
                self.conv_layers.append(Z)

                if i != self.S.num_conv_blocks - 1:
                    Z = self.add_max_pool(Z)

        for i in reversed(range(self.S.num_conv_blocks - 1)):
            num_channels = self.S.num_conv_channels * (2 ** i) * 2

            with tf.variable_scope("deconv%d" % i):
                Z = self.add_deconv_block(
                    Z, self.conv_layers[i], channels=num_channels)
                self.deconv_layers.append(Z)

        Z = self.add_dense_layer("Output", Z, last=True)
        self.dense_layers.append(Z)

    def add_optimizer(self):
        DHW = self.S.image_depth * self.S.image_height * self.S.image_width
        Z = self.dense_layers[-1]

        y_flat = tf.reshape(self.y, [-1])
        y_one_hot_flat = tf.one_hot(y_flat, self.S.num_classes)

        class_weights = tf.constant(
            np.array(self.S.class_weights, dtype=np.float32))
        logging.info("class_weights = %s" % str(class_weights))

        y_weights_flat = tf.reduce_sum(
            tf.multiply(class_weights, y_one_hot_flat), axis=1)
        logging.info("y_weights_flat: %s" % str(y_weights_flat))

        scores = tf.reshape(Z, [-1, self.S.num_classes])

        if self.S.loss == "softmax":
            softmax_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=y_one_hot_flat, logits=scores)
            logging.info("softmax_loss: %s" % str(softmax_loss))
            tf.summary.scalar("softmax_loss", tf.reduce_mean(softmax_loss))

            softmax_weighted_loss = tf.reduce_mean(
                tf.multiply(softmax_loss, y_weights_flat))
            tf.summary.scalar("softmax_weighted_loss", softmax_weighted_loss)

            logging.info("softmax loss selected")
            self.loss += softmax_weighted_loss
        elif self.S.loss == "iou":
            probs = tf.nn.softmax(scores)

            self.loss += iou_op(probs, y_one_hot_flat)

            logging.info("iou loss selected")
        else:
            raise "Unknown loss selected: " + self.S.loss

        tf.summary.scalar("loss", self.loss)

        self.train_step = tf.train.AdamOptimizer(
            learning_rate=self.S.learning_rate).minimize(self.loss)

        self.predictions = tf.cast(tf.argmax(Z, axis=-1), tf.int32)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(self.y, tf.int32), self.predictions), tf.float32))
        self.iou = iou_op(y_one_hot_flat, tf.one_hot(
            tf.reshape(self.predictions, [-1]), self.S.num_classes))

        self.merged_summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(FLAGS.summary,
                                                    self.session.graph)

    def start(self):
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

    def stop(self):
        self.session.close()
        tf.reset_default_graph()

    def weight_variable(self, shape, name):
        init = tflearn.initializations.uniform(minval=-0.05, maxval=0.05)
        return tf.get_variable(name=name, shape=shape, initializer=init)

    def bias_variable(self, shape, name):
        init = tflearn.initializations.zeros()
        return tf.get_variable(name, shape, initializer=init)

    def bias(self, inputs):
        b = self.bias_variable(inputs.shape[-1], "b")
        return inputs + b

    def batch_norm(self, inputs):
        return tf.layers.batch_normalization(inputs, training=self.is_training)

    def batch_norm_or_bias(self, inputs, force_bias=False):
        if force_bias or not self.S.use_batch_norm:
            return self.bias(inputs)

        return self.batch_norm(inputs)

    def dropout(self, inputs):
        return tf.cond(
            self.is_training,
          lambda: tf.nn.dropout(inputs, self.keep_prob),
          lambda: inputs)

    def add_conv_layer(self, Z, kernel_shape=[3, 3, 3], output_channels=None, force_bias=False):
        input_channels = int(Z.shape[4])
        if output_channels is None:
            output_channels = input_channels

        Z = self.dropout(Z)
        logging.info(str(Z))

        W = self.weight_variable(
            kernel_shape + [input_channels, output_channels], "W")
        Z = tf.nn.conv3d(Z, W, [1, 1, 1, 1, 1], padding="SAME")
        logging.info(str(Z))

        Z = self.batch_norm_or_bias(Z, force_bias)
        logging.info(str(Z))

        Z = tf.nn.relu(Z)
        logging.info(str(Z))

        self.loss += 0.5 * self.S.l2_reg * tf.reduce_sum(tf.square(W))

        return Z

    def add_conv_block(self, Z, channels=None):
        depth_factor = 1 if self.S.image_depth == 1 else 3

        with tf.variable_scope("layer1"):
            Z = self.add_conv_layer(
                Z, kernel_shape=[1, 3, 3], output_channels=channels)

        with tf.variable_scope("layer2"):
            Z = self.add_conv_layer(
                Z, kernel_shape=[depth_factor, 1, 3], output_channels=channels)

        with tf.variable_scope("layer3"):
            Z = self.add_conv_layer(
                Z, kernel_shape=[1, 3, 3], output_channels=channels)

        with tf.variable_scope("layer4"):
            Z = self.add_conv_layer(
                Z, kernel_shape=[depth_factor, 3, 1], output_channels=channels)

        return Z

    def add_max_pool(self, Z):
        depth_factor = 2 if self.S.image_depth != 1 else 1

        ksize = [1, depth_factor, 2, 2, 1]
        strides = [1, depth_factor, 2, 2, 1]
        Z = tf.nn.max_pool3d(Z, ksize, strides, "SAME")
        logging.info(str(Z))
        return Z

    def add_deconv_layer(self, Z, output_channels=None):
        batch_size = tf.shape(Z)[0]
        _, input_depth, input_height, input_width, input_channels = Z.shape

        if not output_channels:
            output_channels = input_channels

        Z = self.dropout(Z)
        logging.info(str(Z))

        W = self.weight_variable(
            [1, 1, 1, output_channels, input_channels], "W")

        depth_factor = 2 if self.S.image_depth != 1 else 1

        output_shape = tf.stack([batch_size,
                                 input_depth * depth_factor,
                                 input_height * 2,
                                 input_width * 2,
                                 output_channels])

        Z = tf.nn.conv3d_transpose(
            Z, W, output_shape, [1, depth_factor, 2, 2, 1], padding="SAME")
        logging.info(str(Z))

        Z = self.batch_norm_or_bias(Z)
        logging.info(str(Z))

        Z = tf.nn.relu(Z)
        logging.info(str(Z))

        self.loss += 0.5 * self.S.l2_reg * tf.reduce_sum(tf.square(W))

        return Z

    def add_deconv_block(self, Z, highway_connection, channels=None):
        Z = self.add_deconv_layer(Z, output_channels=channels)
        logging.info(str(Z))

        Z = tf.concat((Z, highway_connection), axis=4)
        logging.info(str(Z))

        Z = self.add_conv_block(Z)

        return Z

    def add_dense_layer(self, name, Z, last):
        output_channels = self.S.num_classes if last else self.S.num_classes

        Z = self.dropout(Z)
        logging.info(str(Z))

        with tf.variable_scope(name):
            W = self.weight_variable(
                [1, 1, 1, int(Z.shape[4]), output_channels], "W")

            Z = tf.nn.conv3d(Z, W, [1, 1, 1, 1, 1], "SAME")
            logging.info(str(Z))

            Z = self.batch_norm_or_bias(Z, force_bias = last)
            logging.info(str(Z))

            if not last:
                Z = tf.nn.relu(Z)
            logging.info(str(Z))

            self.loss += 0.5 * self.S.l2_reg * tf.reduce_sum(tf.square(W))

            return Z

    def fit(self, X, y, step):
        start = timer()
        (_, loss, accuracy, iou, summary) = self.session.run(
            [self.train_step, self.loss, self.accuracy,
                self.iou, self.merged_summary],
          feed_dict={self.X: X, self.y: y, self.is_training: True, self.keep_prob: self.S.keep_prob})
        end = timer()

        if step % 10 == 0:
            logging.info("fit took %.6f sec" % (end - start, ))
            self.summary_writer.add_summary(summary, step)

        return (loss, accuracy, iou)

    def predict(self, X, y=None):
        if y is None:
            (predictions,) = self.session.run(
                [self.predictions],
              feed_dict={self.X: X, self.is_training: False, self.keep_prob: self.S.keep_prob})
            return predictions

        (predictions, loss, accuracy, iou) = self.session.run(
            [self.predictions, self.loss, self.accuracy, self.iou],
            feed_dict={self.X: X, self.y: y, self.is_training: False, self.keep_prob: self.S.keep_prob})
        return (predictions, loss, accuracy, iou)

    def segment(self, image):
        def predictor(X):
            X = np.expand_dims(X, axis=0)
            y = self.predict(X)
            y = np.squeeze(y, axis=0)
            return y

        segmenter = Segmenter(predictor,
                              self.S.image_depth,
                              self.S.image_height,
                              self.S.image_width,
                              image)
        return segmenter.predict()

    def read(self, filepath):
        self.saver.restore(self.session, filepath)
        logging.info("Model restored from file: %s." % filepath)

    def write(self, filepath):
        self.saver.save(self.session, filepath)
        logging.info("Model saved to file: %s." % filepath)


class TestVNet(unittest.TestCase):

    def test_overfit_softmax(self):
        D = 4

        settings = VNet.Settings()
        settings.num_classes = 2
        settings.image_height = settings.image_depth = settings.image_width = D
        settings.num_conv_blocks = 1
        settings.learning_rate = 0.01
        settings.loss = "softmax"
        settings.keep_prob = 1.0
        settings.l2_reg = 0.0

        model = VNet(settings)
        model.add_layers()
        model.add_optimizer()
        model.start()

        X = np.random.randn(1, D, D, D)
        y = (np.random.randn(1, D, D, D) > 0.5).astype(np.uint8)

        X[:, :, :, :] -= .5 * y

        accuracy = 0.0
        for i in range(100):
            loss, accuracy, iou = model.fit(X, y, i)
            logging.info("step %d: loss = %f, accuracy = %f, iou = %f" %
                         (i, loss, accuracy, iou))

        model.stop()

        assert accuracy > 0.95

    def test_metrics(self):
        D = 4
        batch_size = 10

        settings = VNet.Settings()
        settings.num_classes = 2
        settings.class_weights = [1] * settings.num_classes
        settings.image_height = settings.image_depth = settings.image_width = D
        settings.num_conv_blocks = 1
        settings.num_conv_channels = 40
        settings.num_dense_channels = 40
        settings.learning_rate = 0

        model = VNet(settings)
        model.add_layers()
        model.add_optimizer()
        model.start()

        for i in range(10):
            X = np.random.randn(batch_size, D, D, D)
            y = np.random.randint(0, 2, (batch_size, D, D, D), dtype=np.uint)

            y_pred, loss, accuracy, iou = model.predict(X, y)

            accuracy2 = util.accuracy(y_pred, y)
            iou2 = util.iou(y_pred, y, settings.num_classes)

            logging.info("batch %d: loss = %f, accuracy = %f, accuracy2 = %f, iou = %f, iou2 = %f" %
                         (i, loss, accuracy, accuracy2, iou, iou2))

            assert abs(accuracy - accuracy2) < 0.001, "accuracy mismatch!"
            assert abs(iou - iou2) < 0.01, "iou mismatch!"

        model.stop()

    def test_segment_image(self):
        settings = VNet.Settings()
        settings.image_depth = 4
        settings.image_height = 8
        settings.image_width = 6
        settings.num_conv_blocks = 2
        settings.learning_rate = 0.01
        settings.loss = "softmax"
        settings.keep_prob = 0.9
        settings.l2_reg = 0.001
        settings.use_batch_norm = True

        model = VNet(settings)
        model.add_layers()
        model.add_optimizer()
        model.start()

        D, H, W = settings.image_depth, settings.image_height, settings.image_width

        accuracy = 0.0
        iou = 0.0
        for i in range(200):
            X = np.random.randn(10, D, H, W)
            y = (np.random.randn(10, D, H, W) > 0.5).astype(np.uint8)
            X[:, :, :, :] += 10. * y - 5.

            loss, accuracy, iou = model.fit(X, y, i)
            if i % 25 == 24:
                logging.info("batch %d: loss = %f, accuracy = %f, iou = %f" %
                             (i, loss, accuracy, iou))

        D, H, W = settings.image_depth * 2 + \
            3, settings.image_height * 2 + 3, settings.image_width * 2 + 3
        X = np.random.randn(D, H, W)
        y = (np.random.randn(D, H, W) > 0.5).astype(np.uint8)
        X[:, :, :] += 10. * y - 5.

        y_pred = model.segment(X)
        y = (X > 0).astype(np.uint8)

        accuracy2 = util.accuracy(y_pred, y)
        iou2 = util.iou(y_pred, y, settings.num_classes)
        logging.info("segmentation: accuracy = %f, iou = %f" %
                     (accuracy2, iou2))

        model.stop()

        assert abs(accuracy - accuracy2) < 0.1
        assert abs(iou - iou2) < 0.2


if __name__ == '__main__':
    FLAGS(sys.argv)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename='/dev/stderr',
                        filemode='w')
    unittest.main()
