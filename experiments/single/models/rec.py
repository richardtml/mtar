""" rec.py

Simple Recurrent model for Action Recognition.
"""


import tensorflow as tf
from tensorflow.keras import layers


class ARRec(tf.keras.Model):

  def __init__(self, cfg, verbose=False):
    super(ARRec, self).__init__()
    self.verbose = verbose
    rnn = layers.GRU if cfg.rec_type == 'gru' else layers.LSTM
    self.rec = rnn(units=cfg.rec_size)
    self.fc = layers.Dense(101, activation='softmax')

  def call(self, x):

    if self.verbose:
      print(f'x {x.shape}')

    # (N, 16, 512) =>
    # (N, C)
    x = self.rec(x)
    if self.verbose:
      print(f'rec {x.shape}')

    # (N, C) =>
    # (N, 101)
    x = self.fc(x)
    if self.verbose:
      print(f'fc {x.shape}')

    return x
