""" conv2d.py

Simple Conv2D model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers

from experiments.multi.models.shared import HMDB51UCF101


class Conv2D(HMDB51UCF101):

  def __init__(self, cfg, verbose=False):
    super(Conv2D, self).__init__()
    self.verbose = verbose
    self.pad = layers.ZeroPadding2D(padding=(1, 0))
    self.conv2d = layers.Conv2D(
        filters=cfg.conv2d_filters, kernel_size=(3, cfg.reps_size),
        padding='valid', activation='relu')
    self.gap = layers.GlobalAveragePooling1D()
    self.flat = layers.Flatten()

  def call_shared(self, x, verbose):
    if verbose:
      print(f'x {x.shape}')

    # (N, 16, R) =>
    # (N, 16, R, 1)
    x = tf.expand_dims(x, 3)
    if verbose:
      print(f'expand_dims {x.shape}')

    # (N, 16, R, 1) =>
    # (N, 18, R, 1)
    x = self.pad(x)
    if verbose:
      print(f'pad {x.shape}')

    # (N, 16, R, 1) =>
    # (N, 16, 1, F)
    x = self.conv2d(x)
    if verbose:
      print(f'conv2d {x.shape}')

    # (N, 16, 1, F) =>
    # (N, 16, F)
    x = tf.squeeze(x, 2)
    if verbose:
      print(f'squeeze {x.shape}')

    # # (N, 16, F/2) =>
    # # (N, 8, F/2)
    # x = self.pool(x)
    # if self.verbose:
    #   print(f'pool {x.shape}')

    # (N, 8, F/2) =>
    # (N, F/2)
    x = self.gap(x)
    if verbose:
      print(f'gap {x.shape}')

    return x