
import sys
import logging
import unittest
import numpy as np
import tensorflow as tf
import tflearn
import gflags
from timeit import default_timer as timer
import metrics
import util
from segmenter import Segmenter


gflags.DEFINE_string("summary", "./summary/", "")

FLAGS = gflags.FLAGS


class VNet:

    class Settings:
        image_depth = 1
        image_height = 224
        image_width = 224

        num_classes = 2
        class_weights = [1, 1]

        num_conv_blocks = 2
        num_layers_in_conv_block = 4
        num_conv_channels = 16

        num_dense_layers = 0

        learning_rate = 1.0e-4
        lr_decay_steps = 50000
        lr_decay_rate = 0.95
        momentum = 0.9
        clip_gradients = 1.0

        l2_reg = 1e-5
        keep_prob = 0.9

        loss = "softmax"

        # WARNING. Most probably the problem is in batch norm if it shows good
        # perofrmance on training set and fails on validation set miserably.
        use_batch_norm = False

        use_adam_optimizer = False

    def __init__(self, settings):
        self.S = settings

        self.session = tf.Session(
            config=tf.ConfigProto(log_device_placement=False,))

        self.X = tf.placeholder(
            tf.float32, shape=[None, None, None, None])
        self.y = tf.placeholder(
            tf.uint8, shape=[None, None, None, None])
        logging.info("X: %s" % str(self.X))
        logging.info("y: %s" % str(self.y))

        self.step = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)

        self.ifplanar = lambda x, y: x if self.S.image_depth == 1 else y

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
                    Z = self.add_downsample(Z)

        for i in reversed(range(self.S.num_conv_blocks - 1)):
            num_channels = self.S.num_conv_channels * (2 ** i)

            with tf.variable_scope("deconv%d" % i):
                Z = self.add_deconv_block(
                    Z, self.conv_layers[i], channels=num_channels)
                self.deconv_layers.append(Z)

        for i in range(self.S.num_dense_layers):
            Z = self.add_dense_layer("dense%d" % i, Z)
            self.dense_layers.append(Z)
            logging.info(str(Z))

        Z = self.dropout(Z)
        logging.info(str(Z))

        Z = self.add_dense_layer("Output", Z, output_layer=True)
        self.dense_layers.append(Z)
        logging.info(str(Z))

    def add_softmax_loss(self):
        Z = self.dense_layers[-1]

        class_weights = tf.constant(
            np.array(self.S.class_weights, dtype=np.float32))
        logging.info("class_weights = %s" % str(class_weights))

        y_flat = tf.reshape(self.y, [-1])
        y_one_hot_flat = tf.one_hot(y_flat, self.S.num_classes)

        scores = tf.reshape(Z, [-1, self.S.num_classes])

        softmax_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_one_hot_flat, logits=scores)
        logging.info("softmax_loss: %s" % str(softmax_loss))
        tf.summary.scalar("softmax_loss", tf.reduce_mean(softmax_loss))

        y_weights_flat = tf.reduce_sum(
            tf.multiply(class_weights, y_one_hot_flat), axis=1)
        logging.info("y_weights_flat: %s" % str(y_weights_flat))

        softmax_weighted_loss = tf.reduce_mean(
            tf.multiply(softmax_loss, y_weights_flat))
        tf.summary.scalar("softmax_weighted_loss", softmax_weighted_loss)

        logging.info("softmax loss selected")
        self.loss += softmax_weighted_loss

    def add_iou_loss(self):
        Z = self.dense_layers[-1]

        class_weights = tf.constant(
            np.array(self.S.class_weights, dtype=np.float32))
        logging.info("class_weights = %s" % str(class_weights))

        batch_size = tf.shape(Z)[0]

        y_flat = tf.reshape(self.y, [batch_size, -1])
        y_one_hot_flat = tf.one_hot(y_flat, self.S.num_classes)

        scores = tf.reshape(Z, [batch_size, -1, self.S.num_classes])

        probs = tf.nn.softmax(scores)

        iou = metrics.iou_op(probs, y_one_hot_flat)
        logging.info("iou: %s" % str(iou))

        iou_weighted_loss = -tf.reduce_mean(iou * class_weights)
        logging.info("iou_weighted_loss: %s" % str(iou_weighted_loss))

        tf.summary.scalar("iou_weighted_loss", iou_weighted_loss)

        logging.info("iou loss selected")
        self.loss += iou_weighted_loss

    def add_hinge_loss(self):
        Z = self.dense_layers[-1]

        y_flat = tf.reshape(self.y, [-1])
        y_one_hot_flat = tf.one_hot(y_flat, self.S.num_classes)

        class_weights = tf.constant(
            np.array(self.S.class_weights, dtype=np.float32))
        logging.info("class_weights = %s" % str(class_weights))

        y_class_freqs = tf.reduce_sum(y_one_hot_flat, axis=0)
        logging.info("y_class_freqs = %s" % str(y_class_freqs))

        y_class_freq_weights = tf.reshape(
            class_weights * y_class_freqs, [1, self.S.num_classes])
        logging.info("y_class_freq_weights = %s" % str(y_class_freq_weights))

        scores = tf.reshape(Z, [-1, self.S.num_classes])

        hinge_loss = tf.losses.hinge_loss(
            labels=y_one_hot_flat,
            logits=scores,
            weights=y_class_freq_weights)
        logging.info("hinge_loss = %s" % str(hinge_loss))

        tf.summary.scalar("hinge_loss", hinge_loss)

        logging.info("hinge loss selected")
        self.loss += hinge_loss

    def add_optimizer(self):
        if self.S.loss == "softmax":
            self.add_softmax_loss()
        elif self.S.loss == "iou":
            self.add_iou_loss()
        elif self.S.loss == "hinge":
            self.add_hinge_loss()
        else:
            raise ValueError("Unknown loss selected: " + self.S.loss)

        tf.summary.scalar("loss", self.loss)

        if self.S.use_adam_optimizer:
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.S.learning_rate)
        else:
            learning_rate = tf.train.exponential_decay(
                tf.constant(self.S.learning_rate, tf.float32), self.step,
                self.S.lr_decay_steps, self.S.lr_decay_rate)
            self.optimizer = tf.train.MomentumOptimizer(
                learning_rate, self.S.momentum,
                use_nesterov=True)

        if self.S.clip_gradients > 0.0:
            gvs = self.optimizer.compute_gradients(self.loss)
            clipped_gvs = [(tf.clip_by_value(grad, -self.S.clip_gradients, self.S.clip_gradients), var)
                           for grad, var in gvs]
            self.train_step = self.optimizer.apply_gradients(clipped_gvs)
        else:
            self.train_step = self.optimizer.minimize(self.loss)

        self.predictions = tf.cast(
            tf.argmax(self.dense_layers[-1], axis=-1), tf.int32)

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
        variable = tf.get_variable(name=name, shape=shape, initializer=init)
        logging.info(variable)
        return variable

    def bias_variable(self, shape, name):
        init = tflearn.initializations.zeros()
        variable = tf.get_variable(name, shape, initializer=init)
        logging.info(variable)
        return variable

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

    def add_conv_layer(self, Z, kernel_shape=[3, 3, 3], stride=[1, 1, 1], output_channels=None, force_bias=False):
        input_channels = int(Z.shape[4])
        if output_channels is None:
            output_channels = input_channels

        W = self.weight_variable(
            kernel_shape + [input_channels, output_channels], "W")
        Z = tf.nn.conv3d(Z, W, [1] + stride + [1], padding="SAME")
        logging.info(str(Z))

        Z = self.batch_norm_or_bias(Z, force_bias)
        logging.info(str(Z))

        Z = tf.nn.relu(Z)
        logging.info(str(Z))

        self.loss += 0.5 * self.S.l2_reg * tf.reduce_sum(tf.square(W))

        return Z

    def add_conv_block(self, Z, channels=None):
        if channels is None:
            channels = int(Z.shape[4])

        inputs = Z

        for layer in range(self.S.num_layers_in_conv_block):
            with tf.variable_scope("layer%d" % layer):
                Z = self.add_conv_layer(
                    Z, kernel_shape=[1, 3, 3], output_channels=channels)

        Z += inputs

        return Z

    def add_downsample(self, Z):
        Z1 = tf.nn.max_pool3d(Z,
                              ksize=[1, self.ifplanar(1, 2), 2, 2, 1],
                              strides=[1, self.ifplanar(1, 2), 2, 2, 1],
                              padding="SAME")
        logging.info(str(Z1))

        Z2 = self.add_conv_layer(Z,
                                 kernel_shape=[self.ifplanar(1, 3), 3, 3],
                                 stride=[self.ifplanar(1, 2), 2, 2])
        logging.info(str(Z1))

        Z = tf.concat((Z1, Z2), axis=-1)

        return Z

    def add_deconv_layer(self, Z, output_channels=None):
        input_shape = tf.shape(Z)
        batch_size = input_shape[0]
        input_depth = input_shape[1]
        input_height = input_shape[2]
        input_width = input_shape[3]
        input_channels = Z.shape[4]

        if not output_channels:
            output_channels = input_channels

        logging.info("%s %s %s" % (Z, input_channels, output_channels))

        W = self.weight_variable(
            [1, 1, 1, output_channels, input_channels], "W")

        output_shape = tf.stack([batch_size,
                                 input_depth * self.ifplanar(1, 2),
                                 input_height * 2,
                                 input_width * 2,
                                 output_channels])

        Z = tf.nn.conv3d_transpose(
            Z, W, output_shape, [1, self.ifplanar(1, 2), 2, 2, 1], padding="SAME")
        logging.info(str(Z))

        Z = self.batch_norm_or_bias(Z)
        logging.info(str(Z))

        Z = tf.nn.relu(Z)
        logging.info(str(Z))

        self.loss += 0.5 * self.S.l2_reg * tf.reduce_sum(tf.square(W))

        return Z

    def add_deconv_block(self, Z, highway_connection, channels=None):
        if channels is None:
            channels = Z.shape[4]

        logging.info("***** %s %s %s" % (Z, highway_connection, channels))

        Z = self.add_deconv_layer(Z, output_channels=channels)
        logging.info(str(Z))

        Z += highway_connection
        logging.info(str(Z))

        Z = self.add_conv_block(Z)
        logging.info(str(Z))

        return Z

    def add_dense_layer(self, name, Z, output_layer=False):
        output_channels = self.S.num_classes if output_layer else Z.shape[4]

        with tf.variable_scope(name):
            W = self.weight_variable(
                [1, 1, 1, int(Z.shape[4]), output_channels], "W")

            Z = tf.nn.conv3d(Z, W, [1, 1, 1, 1, 1], "SAME")
            logging.info(str(Z))

            Z = self.batch_norm_or_bias(Z, force_bias=output_layer)
            logging.info(str(Z))

            if not output_layer:
                Z = tf.nn.relu(Z)
            logging.info(str(Z))

            self.loss += 0.5 * self.S.l2_reg * tf.reduce_sum(tf.square(W))

            return Z

    def fit(self, X, y, step):
        feed_dict = {
            self.X: X,
            self.y: y,
            self.is_training: True,
            self.keep_prob: self.S.keep_prob,
            self.step: step
        }
        ops = [self.train_step, self.loss,
               self.predictions, self.merged_summary]

        start = timer()
        (_, loss, predictions, summary) = self.session.run(ops, feed_dict)
        end = timer()

        accuracy = metrics.accuracy(y, predictions)
        iou = metrics.iou(y, predictions, self.S.num_classes)

        if step % 10 == 0:
            logging.info("fit took %.6f sec" % (end - start, ))
            self.summary_writer.add_summary(summary, step)

        return (loss, predictions, accuracy, iou)

    def predict(self, X, y=None):
        if y is None:
            (predictions,) = self.session.run(
                [self.predictions],
              feed_dict={self.X: X, self.is_training: False, self.keep_prob: self.S.keep_prob})
            return predictions

        (loss, predictions) = self.session.run(
            [self.loss, self.predictions],
            feed_dict={self.X: X, self.y: y, self.is_training: False, self.keep_prob: self.S.keep_prob})

        accuracy = metrics.accuracy(y, predictions)
        iou = metrics.iou(y, predictions, self.S.num_classes)

        return (loss, predictions, accuracy, iou)

    # If slow = False, then segments image by cutting it to self.S.image_depth x image.height
    # x image.width blocks.
    # If slow = True, then segments image by cutting it to self.S.image_depth x self.S.image_height
    # x self.S.image_width blocks (needs less memory, but slower and makes
    # more mistakes).
    def segment(self, image, slow=False):
        def predictor(X):
            X = np.expand_dims(X, axis=0)
            y = self.predict(X)
            y = np.squeeze(y, axis=0)
            return y

        if slow:
            input_height = self.S.image_height
            input_width = self.S.image_width
        else:
            _, input_height, input_width = image.shape

        segmenter = Segmenter(predictor,
                              self.S.image_depth,
                              input_height,
                              input_width,
                              image)
        return segmenter.predict()

    def read(self, filepath):
        self.saver.restore(self.session, filepath)
        logging.info("Model restored from file: %s." % filepath)

    def write(self, filepath):
        self.saver.save(self.session, filepath)
        logging.info("Model saved to file: %s." % filepath)


class TestVNet(unittest.TestCase):

    def run_overfitting_test(self, loss, size=4, num_conv_blocks=1):
        settings = VNet.Settings()
        settings.num_classes = 2
        if loss == "iou":
            settings.classes_weights = [0.0, 1.0]
        settings.image_height = settings.image_depth = settings.image_width = size
        settings.num_conv_channels = 20
        settings.num_conv_blocks = num_conv_blocks
        settings.learning_rate = 0.2
        settings.loss = loss
        settings.keep_prob = 1.0
        settings.l2_reg = 0.0
        settings.use_batch_norm = True

        model = VNet(settings)
        model.add_layers()
        model.add_optimizer()
        model.start()

        X = np.random.randn(1, size, size, size)
        y = (np.random.randn(1, size, size, size) > 0.5).astype(np.uint8)

        X[:, :, :, :] -= 5.0 * y

        accuracy = 0.0
        iou = 0.0
        for i in range(50):
            loss, predict, accuracy, iou = model.fit(X, y, i)
            if i % 5 == 4:
                logging.info("step %d: loss = %f, accuracy = %f, iou = %f" %
                             (i, loss, accuracy, iou))

        model.stop()

        assert accuracy > 0.95
        assert iou > 0.95

    def test_overfit_softmax(self):
        self.run_overfitting_test(loss="softmax")

    def test_overfit_softmax_2blocks(self):
        self.run_overfitting_test(loss="softmax", num_conv_blocks=2)

    def test_overfit_softmax_3blocks(self):
        self.run_overfitting_test(
            loss="softmax", size=8, num_conv_blocks=3)

    def test_overfit_iou(self):
        self.run_overfitting_test(loss="iou")

    def test_overfit_iou_2blocks(self):
        self.run_overfitting_test(loss="iou", num_conv_blocks=2)

    def test_overfit_iou_3blocks(self):
        self.run_overfitting_test(
            loss="iou", size=8, num_conv_blocks=3)

    def test_overfit_hinge(self):
        self.run_overfitting_test(loss="hinge")

    def test_overfit_hinge_2blocks(self):
        self.run_overfitting_test(loss="hinge", num_conv_blocks=2)

    def test_overfit_hinge_3blocks(self):
        self.run_overfitting_test(
            loss="hinge", size=8, num_conv_blocks=3)

    def test_metrics(self):
        D = 4
        batch_size = 10

        settings = VNet.Settings()
        settings.num_classes = 2
        settings.class_weights = [1] * settings.num_classes
        settings.image_height = settings.image_depth = settings.image_width = D
        settings.num_conv_blocks = 1
        settings.num_conv_channels = 40
        settings.learning_rate = 0

        model = VNet(settings)
        model.add_layers()
        model.add_optimizer()
        model.start()

        for i in range(10):
            X = np.random.randn(batch_size, D, D, D)
            y = np.random.randint(0, 2, (batch_size, D, D, D), dtype=np.uint)

            y_pred, loss, accuracy, iou = model.predict(X, y)

            accuracy2 = metrics.accuracy(y_pred, y)
            iou2 = metrics.iou(y_pred, y, settings.num_classes)

            logging.info("batch %d: loss = %f, accuracy = %f, accuracy2 = %f, iou = %f, iou2 = %f" %
                         (i, loss, accuracy, accuracy2, iou, iou2))

            assert abs(accuracy - accuracy2) < 0.001, "accuracy mismatch!"
            assert abs(iou - iou2) < 0.01, "iou mismatch!"

        model.stop()

    def run_segment_image_test(self, loss="softmax", slow=True):
        settings = VNet.Settings()
        settings.image_depth = 4
        settings.image_height = 8
        settings.image_width = 6
        settings.num_conv_blocks = 2
        settings.num_channels = 16
        settings.learning_rate = 0.1
        settings.loss = loss
        settings.class_weights = [
            1.0, 2.0] if loss == "softmax" else [0.0, 1.0]
        settings.keep_prob = 1.0
        settings.l2_reg = 0.0
        settings.use_batch_norm = False

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

            loss, predict, accuracy, iou = model.fit(X, y, i)
            if i % 25 == 24:
                logging.info("batch %d: loss = %f, accuracy = %f, iou = %f" %
                             (i, loss, accuracy, iou))

        if slow:
            D = settings.image_depth * 2 + 3
            H = settings.image_height * 2 + 3
            W = settings.image_width * 2 + 3
        else:
            D = settings.image_depth * 4
            H = settings.image_height
            W = settings.image_width

        X = np.random.randn(D, H, W)
        y = (np.random.randn(D, H, W) > 0.5).astype(np.uint8)
        X[:, :, :] += 10. * y - 5.

        y_pred = model.segment(X, slow)
        y = (X > 0).astype(np.uint8)

        accuracy2 = metrics.accuracy(y_pred, y)
        iou2 = metrics.iou(y_pred, y, settings.num_classes)
        logging.info("segmentation: accuracy = %f, iou = %f" %
                     (accuracy2, iou2))

        model.stop()

        assert abs(accuracy - accuracy2) < 0.1
        assert abs(iou - iou2) < 0.2

    def test_segment_image_softmax_slow(self):
        self.run_segment_image_test(slow=True)

    def test_segment_image_softmax_fast(self):
        self.run_segment_image_test(slow=False)

    def test_segment_image_iou_slow(self):
        self.run_segment_image_test(slow=True, loss="iou")

    def test_segment_image_iou_fast(self):
        self.run_segment_image_test(slow=False, loss="iou")

if __name__ == '__main__':
    FLAGS(sys.argv)
    util.setup_logging()
    unittest.main()
