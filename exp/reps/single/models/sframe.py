""" mean.py

Simple single frame model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers


class SFrame(tf.keras.Model):

  def __init__(self, cfg, num_classes, verbose=False):
    super(SFrame, self).__init__()
    self.verbose = verbose
    if cfg.ifc:
      self.ifc = layers.Dense(num_classes, activation='relu', name='ifc')
    self.fc = layers.Dense(num_classes, activation='softmax', name='fc')

  def call(self, x, training=False):
    verbose = self.verbose
    # (N, 16, R)
    if verbose: print(f'x {x.shape}')
    # (N, 16, R) => (N, R)
    x = x[:, x.shape[1]//2]
    if verbose: print(f'middle {x.shape}')
    # (N, R) => (N, C)
    if hasattr(self, 'ifc'):
      x = self.ifc(x)
      if verbose: print(f'ifc {x.shape}')
    # (N, R) => (N, C)
    x = self.fc(x)
    # (N, C)
    return x
