""" mean.py

Simple Mean models for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers


class FCMean(tf.keras.Model):

  def __init__(self, cfg, num_classes, verbose=False):
    super(FCMean, self).__init__()
    self.verbose = verbose
    self.fc = layers.Dense(num_classes, activation='softmax', name='fc')
    self.gap = layers.GlobalAveragePooling1D(name='gap')

  def call(self, x, training=False):
    verbose = self.verbose
    shape = x.shape
    # (N, 16, R)
    if verbose: print(f'x {x.shape}')
    # (N, 16, R) => (N*16, R)
    x = tf.reshape(x, (-1, shape[2]))
    if verbose: print(f'reshape {x.shape}')
    # (N*16, R) => (N*16, C)
    x = self.fc(x)
    if verbose: print(f'fc {x.shape}')
    # (N*16, C) => (N, 16, C)
    x = tf.reshape(x, (shape[0], shape[1], -1))
    if verbose: print(f'reshape {x.shape}')
    # (N, 16, C) => (N, C)
    x = self.gap(x)
    if verbose: print(f'gap {x.shape}')
    # (N, C)
    return x


class MeanFC(tf.keras.Model):

  def __init__(self, cfg, num_classes, verbose=False):
    super(MeanFC, self).__init__()
    self.verbose = verbose
    self.gap = layers.GlobalAveragePooling1D(name='gap')
    self.fc = layers.Dense(num_classes, activation='softmax', name='fc')

  def call(self, x, training=False):
    verbose = self.verbose
    # (N, 16, R)
    if verbose: print(f'x {x.shape}')
    # (N, 16, R) => (N, R)
    x = self.gap(x)
    if verbose: print(f'gap {x.shape}')
    # (N, R) => (N, C)
    x = self.fc(x)
    if verbose: print(f'fc {x.shape}')
    # (N, C)
    return x
