""" mean.py

Simple Mean model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers


class Mean(tf.keras.Model):

  def __init__(self, cfg, num_classes, verbose=False):
    super(Mean, self).__init__()
    self.verbose = verbose
    self.gap = layers.GlobalAveragePooling1D(name='gap1d')
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
