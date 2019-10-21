""" rec.py

Shared base model for Action Recognition.
"""


import tensorflow as tf
from tensorflow.keras import layers


class HMDB51UCF101(tf.keras.Model):

  def __init__(self, batchnorm, verbose=False):
    super(HMDB51UCF101, self).__init__()
    self.verbose = verbose
    self.use_batchnorm = batchnorm
    if self.use_batchnorm:
      self.batchnorm = layers.BatchNormalization(name='batchnorm')
    self.fc_hmdb51 = layers.Dense(51, activation='softmax', name='fc_hmdb51')
    self.fc_ucf101 = layers.Dense(101, activation='softmax', name='fc_ucf101')

  def call_shared(self, x, training, verbose):
    # (N, 16, R)
    if verbose: print(f'x {x.shape}')
    # (N, 16, R)
    if self.use_batchnorm:
      x = self.batchnorm(x, training)
      if verbose: print(f'batchnorm {x.shape}')
    return x

  def call(self, xs, training=False):
    # (N, 16, R), (N, 16, R)
    x_hmdb51, x_ucf101 = xs[0], xs[1]

    if self.verbose:
      if x_hmdb51 is not None:
        v_hmdb51, v_ucf101 = (True, False)
      else:
        v_hmdb51, v_ucf101 = (False, True)
    else:
      v_hmdb51, v_ucf101 = (False, False)

    if x_hmdb51 is not None:
      # (N, 16, R)
      x_hmdb51 = HMDB51UCF101.call_shared(self, x_hmdb51, training, v_hmdb51)
      # (N, F)
      x_hmdb51 = self.call_shared(x_hmdb51, training, v_hmdb51)
      # (N, F) =>
      # (N, 51)
      x_hmdb51 = self.fc_hmdb51(x_hmdb51)
      if self.verbose: print(f'fc_hmdb51 {x_hmdb51.shape}')
    if x_ucf101 is not None:
      # (N, 16, R)
      x_ucf101 = HMDB51UCF101.call_shared(self, x_ucf101, training, v_ucf101)
      # (N, F)
      x_ucf101 = self.call_shared(x_ucf101, training, v_ucf101)
      # (N, F) =>
      # (N, 101)
      x_ucf101 = self.fc_ucf101(x_ucf101)
      if self.verbose: print(f'fc_ucf101 {x_ucf101.shape}')
    return x_hmdb51, x_ucf101
