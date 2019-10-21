""" rec.py

Shared base model for Action Recognition.
"""


import tensorflow as tf
from tensorflow.keras import layers


class HMDB51UCF101(tf.keras.Model):

  def __init__(self, verbose=False):
    super(HMDB51UCF101, self).__init__()
    self.verbose = verbose
    self.fc_hmdb51 = layers.Dense(51, activation='softmax')
    self.fc_ucf101 = layers.Dense(101, activation='softmax')

  def call_shared(self, x, verbose):
    # (N, 16, 512)
    if verbose:
      print(f'x {x.shape}')
    return x

  def call(self, xs):
    x_hmdb51, x_ucf101 = xs

    if self.verbose:
      if x_hmdb51 is not None:
        v_hmdb51, v_ucf101 = (True, False)
      else:
        v_hmdb51, v_ucf101 = (False, True)
    else:
      v_hmdb51, v_ucf101 = (False, False)

    if x_hmdb51 is not None:
      # (N, M)
      x_hmdb51 = self.call_shared(x_hmdb51, v_hmdb51)
      # (N, M) =>
      # (N, 51)
      x_hmdb51 = self.fc_hmdb51(x_hmdb51)
      if self.verbose:
        print(f'fc_hmdb51 {x_hmdb51.shape}')

    if x_ucf101 is not None:
      # (N, M)
      x_ucf101 = self.call_shared(x_ucf101, v_ucf101)
      # (N, M) =>
      # (N, 51)
      x_ucf101 = self.fc_ucf101(x_ucf101)
      if self.verbose:
        print(f'fc_ucf101 {x_ucf101.shape}')

    return x_hmdb51, x_ucf101
