""" conv.py

Simple Conv model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers

from exp.reps.multi.models.shared import HMDB51UCF101


class FullConv(HMDB51UCF101):

  def __init__(self, cfg, verbose=False):
    super(FullConv, self).__init__(cfg.batchnorm, verbose)
    div = 1
    self.pad = layers.ZeroPadding2D(padding=(1, 0), name='zpad2d')
    self.conv2d = layers.Conv2D(
        filters=cfg.conv2d_filters, kernel_size=(3, cfg.reps_size),
        padding='valid', activation='relu', name='conv2d')
    self.dropout1 = layers.SpatialDropout1D(cfg.dropout, name='do1d1')
    self.conv1d1 = layers.Conv1D(
        filters=cfg.conv2d_filters//div, kernel_size=3,
        padding='same', activation='relu', name='conv1d1')
    self.dropout2 = layers.SpatialDropout1D(cfg.dropout, name='do1d2')
    self.conv1d2 = layers.Conv1D(
        filters=cfg.conv2d_filters//div, kernel_size=3,
        padding='same', activation='relu', name='conv1d2')
    self.dropout3 = layers.SpatialDropout1D(cfg.dropout, name='do1d3')
    self.conv1d3 = layers.Conv1D(
        filters=cfg.conv2d_filters//div, kernel_size=3,
        padding='same', activation='relu', name='conv1d3')
    self.dropout4 = layers.SpatialDropout1D(cfg.dropout, name='do1d4')
    self.pool1d = layers.MaxPool1D(pool_size=2, name='mpool1d')
    self.flat = layers.Flatten(name='flat')

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
    x = self.dropout1(x, training)
    if verbose: print(f'dropout1 {x.shape}')
    # (N, 16, F) => (N, 8, F)
    x = self.pool1d(x)
    if verbose: print(f'pool1d {x.shape}')
    # (N, 8, F) => (N, 8, F/D)
    x = self.conv1d1(x)
    if verbose: print(f'conv1d1 {x.shape}')
    # (N, 8, F/D) => (N, 8, F/D)
    x = self.dropout2(x, training)
    if verbose: print(f'dropout2 {x.shape}')
    # (N, 8, F/D) => (N, 4, F/D)
    x = self.pool1d(x)
    if verbose: print(f'pool1d {x.shape}')
    # (N, 4, F/D) => (N, 4, F/D)
    x = self.conv1d2(x)
    if verbose: print(f'conv1d2 {x.shape}')
    # (N, 4, F/D) => (N, 4, F/D)
    x = self.dropout3(x, training)
    if verbose: print(f'dropout3 {x.shape}')
    # (N, 4, F/D) => (N, 2, F/D)
    x = self.pool1d(x)
    if verbose: print(f'pool1d {x.shape}')
    # (N, 2, F/D) => (N, 2, F/D)
    x = self.conv1d3(x)
    if verbose: print(f'conv1d3 {x.shape}')
    # (N, 2, F/D) => (N, 2, F/D)
    x = self.dropout4(x, training)
    if verbose: print(f'dropout4 {x.shape}')
    # (N, 2, F/D) => (N, 1, F/D)
    x = self.pool1d(x)
    if verbose: print(f'pool1d {x.shape}')
    # (N, 1, F/D) => (N, F/D)
    x = self.flat(x)
    if verbose: print(f'flat {x.shape}')
    # (N, F/D)
    return x
