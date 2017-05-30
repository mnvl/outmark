
import sys
import logging
import unittest
import numpy as np
import tensorflow as tf
import tflearn
import gflags
import util


gflags.DEFINE_string("summary", "./summary/", "")
gflags.DEFINE_boolean("crfcnn_debug", False, "")

FLAGS = gflags.FLAGS


class CRFCNN:

    class Settings:
        image_height = 224
        image_width = 224
        image_channels = 1

        batch_size = 10

        num_classes = 2

        num_conv_layers = 5
        num_conv_channels = 50

        num_fc_layers = 2
        num_fc_channels = 50

        learning_rate = 1e-4
        l2_reg = 1e-5

        use_batch_norm = False
        keep_prob = 0.9

    def __init__(self, settings):
        self.S = settings

        self.session = tf.Session(
            config=tf.ConfigProto(log_device_placement=FLAGS.crfcnn_debug))

        self.X = tf.placeholder(
            tf.float32, shape=[None, self.S.image_height, self.S.image_width, self.S.image_channels])
        self.phi = tf.placeholder(
            tf.int32, shape=[None, self.S.image_height, self.S.image_width])
        self.psi00 = tf.placeholder(
            tf.int32, shape=[None, self.S.image_height, self.S.image_width])
        self.psi01 = tf.placeholder(
            tf.int32, shape=[None, self.S.image_height, self.S.image_width])
        self.psi10 = tf.placeholder(
            tf.int32, shape=[None, self.S.image_height, self.S.image_width])
        self.psi11 = tf.placeholder(
            tf.int32, shape=[None, self.S.image_height, self.S.image_width])

        logging.info("X: %s" % str(self.X))

        self.is_training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)

        self.train_step = 0
        self.loss = 0

    def add_cnn_layers(self):
        Z = self.X

        for conv_layer in range(self.S.num_conv_layers):
            num_channels = self.S.num_conv_channels

            with tf.variable_scope("conv_layer_%d" % conv_layer):
                Z = self.add_conv_layer(Z, output_channels=num_channels)

            logging.info("CNN layer %d: %s" %
                         (conv_layer, str(Z)))

        self.cnn_output = tf.reshape(Z, [-1, self.S.num_conv_channels])
        logging.info("CNN output: %s" % str(self.cnn_output))

    def add_fc_layers(self, labels, num_classes):
        Z = self.cnn_output

        for fc_layer in range(self.S.num_fc_layers):
            if fc_layer == self.S.num_fc_layers - 1:
                output_channels = num_classes
                activation = lambda x: x
            else:
                output_channels = self.S.num_fc_channels
                activation = tf.nn.relu

            with tf.variable_scope("fc_layer_%d" % fc_layer):
                Z = self.add_fc_layer(Z, output_channels, activation)

            logging.info("FC layer %d: %s" %
                         (fc_layer, str(Z)))

        labels_flat = tf.reshape(labels, [-1])

        self.loss += tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = labels_flat, logits = Z)

    def add_layers(self):
        self.add_cnn_layers()

        with tf.variable_scope("phi"):
            self.add_fc_layers(self.phi, self.S.num_classes)

        with tf.variable_scope("psi00"):
            self.add_fc_layers(self.psi00, self.S.num_classes*self.S.num_classes)

        with tf.variable_scope("psi01"):
            self.add_fc_layers(self.psi01, self.S.num_classes*self.S.num_classes)

        with tf.variable_scope("psi10"):
            self.add_fc_layers(self.psi10, self.S.num_classes*self.S.num_classes)

        with tf.variable_scope("psi11"):
            self.add_fc_layers(self.psi11, self.S.num_classes*self.S.num_classes)

    def start(self):
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())

    def stop(self):
        self.session.close()
        tf.reset_default_graph()

    def weight_variable(self, shape, name):
        init = tflearn.initializations.uniform(minval=-0.05, maxval=0.05)
        return tf.get_variable(name=name, shape=shape, initializer=init)

    def bias_variable(self, shape, name):
        init = tflearn.initializations.zeros()
        return tf.get_variable(name, shape, initializer=init)

    def batch_norm(self, inputs):
        if not self.S.use_batch_norm:
            return tf.layers.batch_normalization(inputs, training=self.is_training)

        return inputs

    def dropout(self, inputs):
        return tf.cond(
            self.is_training,
          lambda: tf.nn.dropout(inputs, self.keep_prob),
          lambda: inputs)

    def add_conv_layer(self, Z, output_channels=None, activation=lambda x: x, kernel_size=3):
        input_channels = int(Z.shape[3])
        if output_channels is None:
            output_channels = input_channels

        W = self.weight_variable(
            [kernel_size, kernel_size, input_channels, output_channels], "W")
        b = self.bias_variable([output_channels], "b")

        Z = tf.nn.conv2d(Z, W, [1, 1, 1, 1], padding="SAME") + b

        Z = activation(Z)

        self.loss += 0.5 * self.S.l2_reg * tf.reduce_sum(tf.square(W))

        return Z

    def add_fc_layer(self, Z, output_channels=None, activation=lambda x: x):
        input_channels = int(Z.shape[1])
        if output_channels is None:
            output_channels = input_channels

        W = self.weight_variable([input_channels, output_channels], "W")
        b = self.bias_variable([output_channels], "b")

        Z = tf.matmul(Z, W) + b

        Z = activation(Z)

        self.loss += 0.5 * self.S.l2_reg * tf.reduce_sum(tf.square(W))

        return Z

    def fit(self, X, y, step):
        y = np.expand_dims(y, 4)
        (_, loss, summary) = self.session.run(
            [self.train_step, self.loss],
          feed_dict={self.X: X, self.y: y, self.is_training: True, self.keep_prob: self.S.keep_prob})

        if step % 10:
            self.summary_writer.add_summary(summary, step)

        return (loss, accuracy, iou)

    def predict(self, X, y=None):
        if y is None:
            (predictions,) = self.session.run(
                [self.predictions],
              feed_dict={self.X: X, self.is_training: False, self.keep_prob: self.S.keep_prob})
            return predictions
        else:
            y = np.expand_dims(y, 4)
            (predictions, loss, accuracy, iou) = self.session.run(
                [self.predictions, self.loss, self.accuracy, self.iou],
              feed_dict={self.X: X, self.y: y, self.is_training: False, self.keep_prob: self.S.keep_prob})
            return (predictions, loss, accuracy, iou)

    def read(self, filepath):
        self.saver.restore(self.session, filepath)
        logging.info("Model restored from file: %s." % filepath)

    def write(self, filepath):
        self.saver.save(self.session, filepath)
        logging.info("Model saved to file: %s." % filepath)


class TestCRFCNN(unittest.TestCase):

    def test_overfit(self):
        D = 4

        settings = CRFCNN.Settings()
        settings.num_classes = 2
        settings.batch_size = 1
        settings.image_height = settings.image_width = D
        settings.image_channels = 1
        settings.learning_rate = 0.01

        model = CRFCNN(settings)
        model.add_layers()
        model.start()

        X = np.random.randn(1, D, D, 1)
        y = (np.random.randn(1, D, D) > 0.5).astype(np.uint8)

        X[:, :, :, 0] -= .5 * y

        for i in range(10):
            loss, accuracy, iou = model.fit(X, y, i)
            logging.info("step %d: loss = %f, accuracy = %f" %
                         (i, loss, accuracy))

        model.stop()

if __name__ == '__main__':
    FLAGS(sys.argv)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename='/dev/stderr',
                        filemode='w')
    unittest.main()
