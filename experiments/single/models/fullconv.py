""" conv.py

Simple Conv model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers


class FullConv(tf.keras.Model):

  def __init__(self, cfg, num_classes, verbose=False):
    super(FullConv, self).__init__()
    self.verbose = verbose
    self.pad = layers.ZeroPadding2D(padding=(1, 0))
    self.conv2d = layers.Conv2D(
        filters=cfg.conv2d_filters, kernel_size=(3, cfg.reps_size),
        padding='valid', activation='relu')
    self.conv1d1 = layers.Conv1D(
        filters=cfg.conv2d_filters//2, kernel_size=3,
        padding='same', activation='relu')
    self.conv1d2 = layers.Conv1D(
        filters=cfg.conv2d_filters//2, kernel_size=3,
        padding='same', activation='relu')
    self.conv1d3 = layers.Conv1D(
        filters=cfg.conv2d_filters//2, kernel_size=3,
        padding='same', activation='relu')
    self.conv1d4 = layers.Conv1D(
        filters=cfg.conv2d_filters//2, kernel_size=3,
        padding='same', activation='relu')
    self.pool = layers.MaxPool1D(pool_size=2)
    self.flat = layers.Flatten()
    self.fc = layers.Dense(num_classes, activation='softmax')

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

    # (N, 16, F) =>
    # (N, 16, F)
    x = self.conv1d1(x)
    if self.verbose:
      print(f'conv1d1 {x.shape}')

    # (N, 16, F) =>
    # (N, 8, F)
    x = self.pool(x)
    if self.verbose:
      print(f'pool {x.shape}')

    # (N, 8, F) =>
    # (N, 8, F)
    x = self.conv1d2(x)
    if self.verbose:
      print(f'conv1d2 {x.shape}')

    # (N, 8, F) =>
    # (N, 4, F)
    x = self.pool(x)
    if self.verbose:
      print(f'pool {x.shape}')

    # (N, 4, F) =>
    # (N, 4, F)
    x = self.conv1d3(x)
    if self.verbose:
      print(f'conv1d3 {x.shape}')

    # (N, 4, F) =>
    # (N, 2, F)
    x = self.pool(x)
    if self.verbose:
      print(f'pool {x.shape}')

    # (N, 2, F) =>
    # (N, 2, F)
    x = self.conv1d4(x)
    if self.verbose:
      print(f'conv1d4 {x.shape}')

    # (N, 2, F) =>
    # (N, 1, F)
    x = self.pool(x)
    if self.verbose:
      print(f'pool {x.shape}')

    # (N, 1, F) =>
    # (N, F)
    x = self.flat(x)
    if self.verbose:
      print(f'flat {x.shape}')

    # (N, F/2) =>
    # (N, C)
    x = self.fc(x)
    if self.verbose:
      print(f'fc {x.shape}')

    return x
