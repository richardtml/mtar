""" conv2d1d.py

Simple Conv2D + Conv1D model for Action Recognition.
"""


import tensorflow as tf
from tensorflow.keras import layers

from exp.reps.multi.models.shared import HMDB51UCF101


class Conv2D1D(HMDB51UCF101):

  def __init__(self, cfg, verbose=False):
    super(Conv2D1D, self).__init__(cfg.batchnorm, verbose)
    div = 2
    self.pad = layers.ZeroPadding2D(padding=(1, 0), name='zpad2d')
    self.conv2d = layers.Conv2D(
        filters=cfg.conv2d_filters, kernel_size=(3, cfg.reps_size),
        padding='valid', activation='relu', name='conv2d')
    self.dropout1 = layers.SpatialDropout1D(cfg.dropout, name='do1d1')
    self.conv1d = layers.Conv1D(
        filters=cfg.conv2d_filters//div, kernel_size=3,
        padding='same', activation='relu', name='conv1d')
    self.dropout2 = layers.SpatialDropout1D(cfg.dropout, name='do1d2')
    self.pool = layers.MaxPool1D(pool_size=2, name='mpool1d')
    self.gap = layers.GlobalAveragePooling1D(name='gap1d')

  def call_shared(self, x, training, verbose):
    # (N, 16, R) => (N, 16, R, 1)
    x = tf.expand_dims(x, 3)
    if verbose: print(f'expand_dims {x.shape}')
    # (N, 16, R, 1) => (N, 18, R, 1)
    x = self.pad(x)
    if verbose: print(f'pad {x.shape}')
    # (N, 18, R, 1) => (N, 16, 1, F)
    x = self.conv2d(x)
    if verbose: print(f'conv2d {x.shape}')
    # (N, 16, 1, F) => (N, 16, F)
    x = tf.squeeze(x, 2)
    if verbose: print(f'squeeze {x.shape}')
    # (N, 16, F) => (N, 16, F)
    x = self.dropout(x, training)
    if verbose: print(f'dropout {x.shape}')
    # (N, 16, F) => (N, 8, F)
    x = self.pool(x)
    if verbose: print(f'pool {x.shape}')
    # (N, 8, F) => (N, 8, F/D)
    x = self.conv1d(x)
    if verbose: print(f'conv1d {x.shape}')
    # (N, 8, F/D) => (N, 8, F/D)
    x = self.dropout(x, training)
    if verbose: print(f'dropout {x.shape}')
    # (N, 8, F/D) => (N, F/D)
    x = self.gap(x)
    if verbose: print(f'gap {x.shape}')
    # (N, F/D)
    return x
