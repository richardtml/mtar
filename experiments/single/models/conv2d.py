""" conv2d.py

Simple Conv2D model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers


class ARConv2D(tf.keras.Model):

  def __init__(self, cfg, verbose=False):
    super(ARConv2D, self).__init__()
    self.verbose = verbose
    self.pad = layers.ZeroPadding2D(padding=(1, 0))
    self.conv2d = layers.Conv2D(
        filters=cfg.conv2d_filters, kernel_size=(3, cfg.reps_size),
        padding='valid', activation='relu')
    self.pool = layers.MaxPool1D(pool_size=2)
    self.gap = layers.GlobalAveragePooling1D()
    self.flat = layers.Flatten()
    self.fc = layers.Dense(101, activation='softmax')

  def call(self, x):
    if self.verbose:
      print(f'x {x.shape}')

    # (N, 16, R) =>
    # (N, 16, R, 1)
    x = tf.expand_dims(x, 3)
    if self.verbose:
      print(f'expand_dims {x.shape}')

    # (N, 16, R, 1) =>
    # (N, 18, R, 1)
    x = self.pad(x)
    if self.verbose:
      print(f'pad {x.shape}')

    # (N, 16, R, 1) =>
    # (N, 16, 1, F)
    x = self.conv2d(x)
    if self.verbose:
      print(f'conv2d {x.shape}')

    # (N, 16, 1, F) =>
    # (N, 16, F)
    x = tf.squeeze(x, 2)
    if self.verbose:
      print(f'squeeze {x.shape}')

    # (N, 16, F/2) =>
    # (N, 8, F/2)
    x = self.pool(x)
    if self.verbose:
      print(f'pool {x.shape}')

    # (N, 8, F/2) =>
    # (N, F/2)
    x = self.gap(x)
    if self.verbose:
      print(f'gap {x.shape}')

    # (N, F/2) =>
    # (N, 101)
    x = self.fc(x)
    if self.verbose:
      print(f'fc {x.shape}')

    return x
