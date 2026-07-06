"""Long/short (margin) extension of the EIIE trading system.

Differences from the long-only ``eiie.py`` agent:

* **Direction**: the output head produces a softmax over ``2m + 1`` slots -
  cash, m long slots and m short slots.  The net signed exposure of coin i
  is ``w_long_i - w_short_i``, so the network chooses allocation *and*
  direction jointly and the gross exposure never exceeds 1 before leverage.
* **Leverage**: a separate head (global average over the per-asset feature
  maps, then a dense layer + sigmoid) outputs a leverage multiplier in
  ``[0, max_leverage]`` that scales the whole signed portfolio.
* **Per-coin cap**: signed weights are clipped to ``+-max_coin_weight``
  after leverage, both in training and in trading.
* **USDT accounting**: equity, PnL, financing and commissions are all in
  USDT (quote currency).  Per period the equity return is

      R_t = 1 + sum_i w_i (y_i - 1)
              - c * turnover
              - borrow costs (shorted coins + borrowed USDT)

  where ``w_i`` is the signed fraction of equity in coin i.  Shorting a
  coin borrows the coin (financing at ``short_borrow_apr``); long exposure
  beyond the equity borrows USDT (financing at ``usdt_borrow_apr``).
* **Liquidation guard**: if equity would drop below the maintenance floor
  the account is stopped out (this is a feature of the simulator; the
  training loss floors R at a small epsilon so gradients stay finite).

The risk overlays (volatility targeting and drawdown-based deleveraging)
live in the backtester (`margin_backtest.py`), not in the network.
"""

from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import tensorflow as tf

from pgportfolio.dynamic.eiie import PVMDataSource

DAY = 86400.0


class MarginEIIECore(tf.Module):
    """EIIE convolution stack with a long/short/cash head and a leverage head."""

    def __init__(self, feature_number=3, window_size=31,
                 conv_filters=3, dense_filters=10,
                 max_leverage=1.0, max_coin_weight=1.0,
                 dense_weight_decay=5e-9, output_weight_decay=5e-8):
        super(MarginEIIECore, self).__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=100)
        self.conv1_kernel = tf.Variable(
            initializer([1, 2, feature_number, conv_filters]), name="conv1_kernel")
        # small positive bias: normalized price inputs are ~1.0 and zero
        # biases leave relu units dead (see eiie.EIIECore)
        self.conv1_bias = tf.Variable(0.1 * tf.ones([conv_filters]), name="conv1_bias")
        self.conv2_kernel = tf.Variable(
            initializer([1, window_size - 1, conv_filters, dense_filters]),
            name="conv2_kernel")
        self.conv2_bias = tf.Variable(0.1 * tf.ones([dense_filters]), name="conv2_bias")
        # two logits per coin: long and short
        self.conv3_kernel = tf.Variable(
            initializer([1, 1, dense_filters + 1, 2]), name="conv3_kernel")
        self.conv3_bias = tf.Variable(tf.zeros([2]), name="conv3_bias")
        self.cash_bias = tf.Variable(tf.zeros([1, 1]), name="cash_bias")
        # leverage head: dense on the globally averaged feature map
        self.lev_kernel = tf.Variable(
            initializer([dense_filters + 1, 1]), name="lev_kernel")
        self.lev_bias = tf.Variable(tf.zeros([1]), name="lev_bias")
        self._max_leverage = max_leverage
        self._max_coin_weight = max_coin_weight
        self._dense_weight_decay = dense_weight_decay
        self._output_weight_decay = output_weight_decay

    def __call__(self, x, previous_w):
        """
        :param x: [batch, features, assets, window]
        :param previous_w: [batch, assets] previous *signed* weights
        :return: [batch, assets] signed portfolio weights (fraction of
            equity; cash is the residual 1 - sum(max(w,0)) + sum(max(-w,0)))
        """
        network = tf.transpose(x, [0, 2, 3, 1])
        # centered returns + leaky_relu: see eiie.EIIECore.__call__
        # features 0..2 are prices (close/high/low), normalized by the last
        # close; an optional feature 3 is volume, normalized by its own
        # window mean (a relative-activity signal)
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
                                   + self.conv2_bias, alpha=0.01)  # [b, m, 1, F]
        w = previous_w[:, :, None, None]
        features = tf.concat([network, w], axis=3)       # [b, m, 1, F+1]

        logits = tf.nn.conv2d(features, self.conv3_kernel,
                              strides=1, padding="VALID") + self.conv3_bias
        long_logits = logits[:, :, 0, 0]
        short_logits = logits[:, :, 0, 1]
        cash = tf.tile(self.cash_bias, [tf.shape(long_logits)[0], 1])
        softmax = tf.nn.softmax(
            tf.concat([cash, long_logits, short_logits], axis=1))
        assets = tf.shape(long_logits)[1]
        w_long = softmax[:, 1:assets + 1]
        w_short = softmax[:, assets + 1:]
        net = w_long - w_short                            # gross <= 1

        pooled = tf.reduce_mean(features[:, :, 0, :], axis=1)  # [b, F+1]
        leverage = self._max_leverage * tf.nn.sigmoid(
            tf.matmul(pooled, self.lev_kernel) + self.lev_bias)  # [b, 1]
        signed = leverage * net
        signed = tf.clip_by_value(signed, -self._max_coin_weight,
                                  self._max_coin_weight)
        return signed, softmax

    def regularization_loss(self):
        return (self._dense_weight_decay * tf.nn.l2_loss(self.conv2_kernel) +
                self._output_weight_decay * tf.nn.l2_loss(self.conv3_kernel) +
                self._output_weight_decay * tf.nn.l2_loss(self.lev_kernel))


def margin_period_return(w, y, previous_w, commission_rate,
                         short_borrow_per_period, usdt_borrow_per_period):
    """Equity relative return of one period for signed weights (numpy).

    :param w: [m] signed target weights (fraction of equity)
    :param y: [m] price relatives close_{t+1}/close_t
    :param previous_w: [m] signed weights held before rebalancing
    :return: (R, drifted weights after the period)
    """
    turnover = np.abs(w - previous_w).sum()
    long_exposure = np.maximum(w, 0).sum()
    short_exposure = np.maximum(-w, 0).sum()
    usdt_borrowed = max(0.0, long_exposure - 1.0)
    R = (1.0 + np.dot(w, y - 1.0)
         - commission_rate * turnover
         - short_exposure * short_borrow_per_period
         - usdt_borrowed * usdt_borrow_per_period)
    if R <= 0:
        return 0.0, np.zeros_like(w)
    drifted = w * y / R
    return R, drifted


class MarginEIIEAgent(object):
    def __init__(self, feature_number=3, window_size=31,
                 commission_rate=0.001, learning_rate=0.00028,
                 max_leverage=1.0, max_coin_weight=1.0,
                 short_borrow_apr=0.10, usdt_borrow_apr=0.10,
                 trade_period=1800, gross_penalty=0.0,
                 boundary_penalty=1e-4, conv_filters=3, dense_filters=10):
        self._window = window_size
        self._commission = commission_rate
        self._boundary_penalty = boundary_penalty
        periods_per_day = DAY / trade_period
        self._short_borrow = short_borrow_apr / 365.0 / periods_per_day
        self._usdt_borrow = usdt_borrow_apr / 365.0 / periods_per_day
        self._gross_penalty = gross_penalty
        self.net = MarginEIIECore(feature_number=feature_number,
                                  window_size=window_size,
                                  max_leverage=max_leverage,
                                  max_coin_weight=max_coin_weight,
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
        w, softmax = self.net(x, last_w)                 # signed [b, m]
        price_change = y[:, 0, :] - 1.0
        turnover = tf.reduce_sum(tf.abs(w - last_w), axis=1)
        long_exposure = tf.reduce_sum(tf.nn.relu(w), axis=1)
        short_exposure = tf.reduce_sum(tf.nn.relu(-w), axis=1)
        usdt_borrowed = tf.nn.relu(long_exposure - 1.0)
        R = (1.0 + tf.reduce_sum(w * price_change, axis=1)
             - self._commission * turnover
             - self._short_borrow * short_exposure
             - self._usdt_borrow * usdt_borrowed)
        R = tf.maximum(R, 1e-2)  # keep log finite through liquidations
        loss = -tf.reduce_mean(tf.math.log(R))
        if self._gross_penalty:
            loss += self._gross_penalty * tf.reduce_mean(
                long_exposure + short_exposure)
        if self._boundary_penalty:
            # keep the long/short/cash softmax away from saturated corners
            # (same role as loss_function7's LAMBDA term, see eiie.py)
            loss += self._boundary_penalty * tf.reduce_mean(
                tf.reduce_sum(-tf.math.log(1 + 1e-6 - softmax), axis=1))
        return loss + self.net.regularization_loss(), w

    def _train_step_impl(self, x, y, last_w):
        with tf.GradientTape() as tape:
            loss, w = self._loss(x, y, last_w)
        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self._optimizer.apply_gradients(zip(gradients, variables))
        return loss, w

    def train_on_panel(self, panel, steps, batch_size=109, sample_bias=5e-5,
                       log_every=5000, tag=""):
        source = PVMDataSource(panel, self._window, batch_size, sample_bias)
        # signed weights: PVM starts flat at 0 (all cash) instead of 1/m
        source._pvm = np.zeros_like(source._pvm)
        running = []
        for step in range(steps):
            x, y, last_w, setw = source.next_batch()
            loss, w = self._train_step(
                tf.constant(x), tf.constant(y), tf.constant(last_w))
            setw(np.clip(w.numpy(), -1.0, 1.0))
            running.append(float(loss))
            if log_every and (step + 1) % log_every == 0:
                logging.info("%s step %d/%d loss %.6f", tag, step + 1, steps,
                             np.mean(running[-log_every:]))
        return np.mean(running[-min(len(running), 1000):]) if running else None

    def decide_by_history(self, history, last_w):
        """:param last_w: [assets] previous signed weights (no cash slot).
        :return: [assets] signed target weights"""
        w, _ = self._forward(
            tf.constant(history[None, ...], dtype=tf.float32),
            tf.constant(last_w[None, :], dtype=tf.float32))
        return np.squeeze(w.numpy(), axis=0)
