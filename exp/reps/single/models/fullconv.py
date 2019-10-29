""" conv.py

Simple Conv model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers


def print_dims(layer, shape, verbose):
  if verbose:
    print(f'{layer} {shape}')


class FullConv(tf.keras.Model):

  def __init__(self, cfg, num_classes, verbose=False):
    super(FullConv, self).__init__()
    self.verbose = verbose
    div = 1
    self.pad = layers.ZeroPadding2D(padding=(1, 0), name='zpad2d')
    self.conv2d = layers.Conv2D(
        filters=cfg.conv2d_filters, kernel_size=(3, cfg.reps_size),
        padding='valid', activation='relu', name='conv2d')
    self.dropout = layers.SpatialDropout1D(cfg.dropout, name='do1d')
    self.conv1d1 = layers.Conv1D(
        filters=cfg.conv1d_filters, kernel_size=3,
        padding='same', activation='relu', name='conv1d1')
    self.conv1d2 = layers.Conv1D(
        filters=cfg.conv1d_filters, kernel_size=3,
        padding='same', activation='relu', name='conv1d2')
    self.conv1d3 = layers.Conv1D(
        filters=cfg.conv1d_filters, kernel_size=3,
        padding='same', activation='relu', name='conv1d3')
    self.pool1d = layers.MaxPool1D(pool_size=2, name='mpool1d')
    self.flat = layers.Flatten(name='flat')
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
    x = self.pool1d(x)
    if verbose: print(f'pool1d {x.shape}')
    # (N, 8, F) => (N, 8, F/D)
    x = self.conv1d1(x)
    if verbose: print(f'conv1d1 {x.shape}')
    # (N, 8, F/D) => (N, 8, F/D)
    x = self.dropout(x, training)
    if verbose: print(f'dropout {x.shape}')
    # (N, 8, F/D) => (N, 4, F/D)
    x = self.pool1d(x)
    if verbose: print(f'pool1d {x.shape}')
    # (N, 4, F/D) => (N, 4, F/D)
    x = self.conv1d2(x)
    if verbose: print(f'conv1d2 {x.shape}')
    # (N, 4, F/D) => (N, 4, F/D)
    x = self.dropout(x, training)
    if verbose: print(f'dropout {x.shape}')
    # (N, 4, F/D) => (N, 2, F/D)
    x = self.pool1d(x)
    if verbose: print(f'pool1d {x.shape}')
    # (N, 2, F/D) => (N, 2, F/D)
    x = self.conv1d3(x)
    if verbose: print(f'conv1d3 {x.shape}')
    # (N, 2, F/D) => (N, 2, F/D)
    x = self.dropout(x, training)
    if verbose: print(f'dropout {x.shape}')
    # (N, 2, F/D) => (N, 1, F/D)
    x = self.pool1d(x)
    if verbose: print(f'pool1d {x.shape}')
    # (N, 1, F/D) => (N, F/D)
    x = self.flat(x)
    if verbose: print(f'flat {x.shape}')
    # (N, F/D) => (N, C)
    x = self.fc(x)
    if verbose: print(f'fc {x.shape}')
    # (N, C)
    return x
