""" mean.py

Simple Mean models for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers


class FCMean(tf.keras.Model):

  def __init__(self, cfg, num_classes, verbose=False):
    super(FCMean, self).__init__()
    self.verbose = verbose
    if cfg.hfc:
      self.hfc = layers.Dense(num_classes, activation='relu', name='hfc')
    else:
      self.hfc = None
    self.fc = layers.Dense(num_classes, activation='softmax', name='fc')
    self.gap = layers.GlobalAveragePooling1D(name='gap')

  def call(self, x, training=False):
    verbose = self.verbose
    shape = x.shape
    # (N, F, R)
    if verbose: print(f'x {x.shape}')
    # (N, F, R) => (N*F, R)
    x = tf.reshape(x, (-1, shape[2]))
    if verbose: print(f'reshape {x.shape}')
    # (N*F, R) => (N*F, C)
    if self.hfc:
      x = self.hfc(x)
      if verbose: print(f'hfc {x.shape}')
    # (N*F, R) => (N*F, C)
    x = self.fc(x)
    if verbose: print(f'fc {x.shape}')
    # (N*F, C) => (N, F, C)
    x = tf.reshape(x, (shape[0], shape[1], -1))
    if verbose: print(f'reshape {x.shape}')
    # (N, F, C) => (N, C)
    x = self.gap(x)
    if verbose: print(f'gap {x.shape}')
    # (N, C)
    return x


class MeanFC(tf.keras.Model):

  def __init__(self, cfg, num_classes, verbose=False):
    super(MeanFC, self).__init__()
    self.verbose = verbose
    self.gap = layers.GlobalAveragePooling1D(name='gap')
    if cfg.hfc:
      self.hfc = layers.Dense(num_classes, activation='relu', name='hfc')
    else:
      self.hfc = None
    self.fc = layers.Dense(num_classes, activation='softmax', name='fc')

  def call(self, x, training=False):
    verbose = self.verbose
    # (N, F, R)
    if verbose: print(f'x {x.shape}')
    # (N, F, R) => (N, R)
    x = self.gap(x)
    if verbose: print(f'gap {x.shape}')
    # (N, R) => (N, C)
    if self.hfc:
      x = self.hfc(x)
      if verbose: print(f'hfc {x.shape}')
    # (N, R) => (N, C)
    x = self.fc(x)
    if verbose: print(f'fc {x.shape}')
    # (N, C)
    return x
