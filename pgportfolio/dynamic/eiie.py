"""TensorFlow 2 port of PGPortfolio's EIIE network and trainer.

This is a faithful re-implementation of the default topology in
``pgportfolio/net_config.json`` (see ``pgportfolio/learn/network.py`` and
``nnagent.py`` for the original TF1/tflearn code):

    ConvLayer   (1x2 kernel,  3 filters, relu)
    EIIE_Dense  (1x(w-1) kernel, 10 filters, relu, L2 5e-9)
    EIIE_Output_WithW (previous weights appended as a feature map,
                       1x1 conv, trainable cash bias, softmax)

trained with ``loss_function6``: the negative log of the commission-adjusted
portfolio value over a batch of *consecutive* periods, sampled geometrically
from a portfolio-vector-memory (PVM) replay buffer.

A crucial property of the EIIE ("Ensemble of Identical Independent
Evaluators") topology is that every convolution has kernel height 1 and all
weights are shared across assets, so a trained network is agnostic to both
the *identity* and the *number* of assets.  That is exactly what makes it
usable with a dynamic universe: the same weights keep working when the
top-10 selection changes from month to month.
"""

from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import tensorflow as tf


class EIIECore(tf.Module):
    """The convolutional policy network."""

    def __init__(self, feature_number=3, window_size=31,
                 conv_filters=3, dense_filters=10,
                 dense_weight_decay=5e-9, output_weight_decay=5e-8):
        super(EIIECore, self).__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=100)
        self.conv1_kernel = tf.Variable(
            initializer([1, 2, feature_number, conv_filters]), name="conv1_kernel")
        # normalized price inputs sit close to 1.0, so zero-initialized biases
        # leave many relu units dead from the start (no gradient ever flows);
        # a small positive bias keeps them initially active
        self.conv1_bias = tf.Variable(0.1 * tf.ones([conv_filters]), name="conv1_bias")
        self.conv2_kernel = tf.Variable(
            initializer([1, window_size - 1, conv_filters, dense_filters]),
            name="conv2_kernel")
        self.conv2_bias = tf.Variable(0.1 * tf.ones([dense_filters]), name="conv2_bias")
        self.conv3_kernel = tf.Variable(
            initializer([1, 1, dense_filters + 1, 1]), name="conv3_kernel")
        self.conv3_bias = tf.Variable(tf.zeros([1]), name="conv3_bias")
        self.cash_bias = tf.Variable(tf.zeros([1, 1]), name="cash_bias")
        self._dense_weight_decay = dense_weight_decay
        self._output_weight_decay = output_weight_decay

    def __call__(self, x, previous_w):
        """
        :param x: [batch, features, assets, window] price tensor
        :param previous_w: [batch, assets] weights of the previous period
            (cash excluded)
        :return: [batch, assets+1] portfolio weights (cash first)
        """
        network = tf.transpose(x, [0, 2, 3, 1])          # [b, m, w, f]
        # normalize by the latest close of each asset and center around 0:
        # raw ratios sit at ~1.0, and an all-positive input makes it easy
        # for randomly initialized relu conv channels to be dead for every
        # asset at once (no gradient ever flows); centered returns plus
        # leaky_relu remove that failure mode.  An optional 4th feature is
        # volume, normalized by its own window mean.
        prices = network[:, :, :, :3] / network[:, :, -1:, 0:1] - 1.0
        if network.shape[3] is not None and network.shape[3] > 3:
            volume = network[:, :, :, 3:4]
            volume = volume / (tf.reduce_mean(volume, axis=2, keepdims=True)
                               + 1e-8) - 1.0
            network = tf.concat([prices, tf.clip_by_value(volume, -1.0, 9.0)],
                                axis=3)
        else:
            network = prices
        network = tf.nn.leaky_relu(tf.nn.conv2d(network, self.conv1_kernel,
                                                strides=1, padding="VALID")
                                   + self.conv1_bias, alpha=0.01)
        network = tf.nn.leaky_relu(tf.nn.conv2d(network, self.conv2_kernel,
                                                strides=1, padding="VALID")
                                   + self.conv2_bias, alpha=0.01)  # [b, m, 1, filters]
        w = previous_w[:, :, None, None]                 # [b, m, 1, 1]
        network = tf.concat([network, w], axis=3)
        network = tf.nn.conv2d(network, self.conv3_kernel,
                               strides=1, padding="VALID") + self.conv3_bias
        votes = network[:, :, 0, 0]                      # [b, m]
        cash = tf.tile(self.cash_bias, [tf.shape(votes)[0], 1])
        return tf.nn.softmax(tf.concat([cash, votes], axis=1))

    def regularization_loss(self):
        # tflearn's L2 regularizer contributes weight_decay * sum(W**2) / 2
        return (self._dense_weight_decay * tf.nn.l2_loss(self.conv2_kernel) +
                self._output_weight_decay * tf.nn.l2_loss(self.conv3_kernel))


class PVMDataSource(object):
    """Replay buffer with portfolio vector memory and geometric sampling.

    Mirrors ``DataMatrices``/``ReplayBuffer``: a batch is a run of
    consecutive periods whose start index is drawn from a geometric
    distribution favouring the most recent data; the network's outputs are
    written back into the PVM so the next epoch sees realistic previous
    weights.
    """

    def __init__(self, panel, window_size, batch_size, sample_bias):
        """:param panel: [features, assets, time] price array; feature 0 must
        be the close."""
        self._panel = np.asarray(panel, dtype=np.float32)
        self._window = window_size
        self._batch_size = batch_size
        self._bias = sample_bias
        assets, times = self._panel.shape[1], self._panel.shape[2]
        self._pvm = np.full((times, assets), 1.0 / assets, dtype=np.float32)
        # last valid decision index: needs window history + 1 future price
        self._end = times - window_size - 1
        if self._end - 1 <= batch_size:
            raise ValueError("training window too small: %d usable periods"
                             % self._end)

    def _sample_start(self):
        upper = self._end - self._batch_size
        ran = np.random.geometric(self._bias)
        while ran > upper - 1:
            ran = np.random.geometric(self._bias)
        return upper - ran

    def next_batch(self):
        start = self._sample_start()
        indices = np.arange(start, start + self._batch_size)
        M = np.stack([self._panel[:, :, i:i + self._window + 1]
                      for i in indices])                  # [b, f, m, w+1]
        X = M[:, :, :, :-1]
        y = M[:, :, :, -1] / M[:, 0, None, :, -2]
        last_w = self._pvm[indices - 1, :]

        def setw(w):
            self._pvm[indices, :] = w
        return X, y, last_w, setw


class EIIEAgent(object):
    """NNAgent equivalent: owns the network, the loss and the optimizer."""

    def __init__(self, feature_number=3, window_size=31,
                 commission_rate=0.0025, learning_rate=0.00028,
                 boundary_penalty=1e-4, conv_filters=3, dense_filters=10):
        self._window = window_size
        self._commission = commission_rate
        # loss_function7-style repulsion from the simplex corners (LAMBDA in
        # the original constants.py): without it the shared voting layer can
        # drive every asset's logit down together until the softmax
        # saturates at 100% cash, where all gradients vanish and training
        # freezes permanently.
        self._boundary_penalty = boundary_penalty
        self.net = EIIECore(feature_number=feature_number,
                            window_size=window_size,
                            conv_filters=conv_filters,
                            dense_filters=dense_filters)
        self._optimizer = tf.keras.optimizers.Adam(learning_rate)
        self._train_step = tf.function(
            self._train_step_impl,
            input_signature=[
                tf.TensorSpec([None, feature_number, None, window_size], tf.float32),
                tf.TensorSpec([None, feature_number, None], tf.float32),
                tf.TensorSpec([None, None], tf.float32),
            ])
        self._forward = tf.function(
            self.net.__call__,
            input_signature=[
                tf.TensorSpec([None, feature_number, None, window_size], tf.float32),
                tf.TensorSpec([None, None], tf.float32),
            ])

    def _loss(self, x, y, last_w):
        output = self.net(x, last_w)
        future_price = tf.concat([tf.ones_like(y[:, 0, 0:1]), y[:, 0, :]], axis=1)
        gains = tf.reduce_sum(output * future_price, axis=1)
        future_omega = future_price * output / gains[:, None]
        # commission approximation over consecutive samples (__pure_pc)
        w_t = future_omega[:-1]
        w_t1 = output[1:]
        mu = 1 - tf.reduce_sum(tf.abs(w_t1[:, 1:] - w_t[:, 1:]), axis=1) * \
            self._commission
        pv_vector = gains * tf.concat([tf.ones(1), mu], axis=0)
        loss = -tf.reduce_mean(tf.math.log(pv_vector))
        if self._boundary_penalty:
            loss += self._boundary_penalty * tf.reduce_mean(
                tf.reduce_sum(-tf.math.log(1 + 1e-6 - output), axis=1))
        return loss + self.net.regularization_loss(), output

    def _train_step_impl(self, x, y, last_w):
        with tf.GradientTape() as tape:
            loss, output = self._loss(x, y, last_w)
        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self._optimizer.apply_gradients(zip(gradients, variables))
        return loss, output

    def train_on_panel(self, panel, steps, batch_size=109, sample_bias=5e-5,
                       log_every=5000, tag=""):
        source = PVMDataSource(panel, self._window, batch_size, sample_bias)
        running = []
        for step in range(steps):
            x, y, last_w, setw = source.next_batch()
            loss, output = self._train_step(
                tf.constant(x), tf.constant(y), tf.constant(last_w))
            setw(output.numpy()[:, 1:])
            running.append(float(loss))
            if log_every and (step + 1) % log_every == 0:
                logging.info("%s step %d/%d loss %.6f", tag, step + 1, steps,
                             np.mean(running[-log_every:]))
        return np.mean(running[-min(len(running), 1000):]) if running else None

    def decide_by_history(self, history, last_w):
        """
        :param history: [features, assets, window] price matrix
        :param last_w: [assets+1] current portfolio (cash first)
        :return: [assets+1] target portfolio weights
        """
        output = self._forward(
            tf.constant(history[None, ...], dtype=tf.float32),
            tf.constant(last_w[None, 1:], dtype=tf.float32))
        return np.squeeze(output.numpy(), axis=0)
