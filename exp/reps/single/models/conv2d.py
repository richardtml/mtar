""" conv2d.py

Simple Conv2D model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers


class Conv2D(tf.keras.Model):

  def __init__(self, cfg, num_classes, verbose=False):
    super(Conv2D, self).__init__()
    self.verbose = verbose
    self.pad = layers.ZeroPadding2D(padding=(1, 0), name='zpad2d')
    self.conv2d = layers.Conv2D(
        filters=cfg.conv2d_filters, kernel_size=(3, cfg.reps_size),
        padding='valid', activation='relu', name='conv2d')
    self.dropout = layers.SpatialDropout1D(cfg.dropout, name='do1d')
    self.gap = layers.GlobalAveragePooling1D(name='gap1d')
    self.fc = layers.Dense(num_classes, activation='softmax', name='fc')

  def call(self, x, training=False):
    verbose = self.verbose
    # (N, 16, R)
    if verbose: print(f'x {x.shape}')
    # (N, 16, R) => (N, 16, R, 1)
    x = tf.expand_dims(x, 3)
    if verbose: print(f'expand_dims {x.shape}')
    # (N, 16, R, 1) => (N, 18, R, 1)
    x = self.pad(x)
    if verbose: print(f'pad {x.shape}')
    # (N, 16, R, 1) => (N, 16, 1, F)
    x = self.conv2d(x)
    if verbose: print(f'conv2d {x.shape}')
    # (N, 16, 1, F) => (N, 16, F)
    x = tf.squeeze(x, 2)
    if verbose: print(f'squeeze {x.shape}')
    # (N, 16, F) => (N, 16, F)
    x = self.dropout(x, training)
    if verbose: print(f'dropout {x.shape}')
    # (N, 8, F) => (N, F)
    x = self.gap(x)
    if verbose: print(f'gap {x.shape}')
    # (N, F) => (N, C)
    x = self.fc(x)
    if verbose: print(f'fc {x.shape}')
    # (N, C)
    return x
