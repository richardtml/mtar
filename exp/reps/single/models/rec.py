""" rec.py

Simple Recurrent model for Action Recognition.
"""


import tensorflow as tf
from tensorflow.keras import layers


class Rec(tf.keras.Model):

  def __init__(self, cfg, num_classes, verbose=False):
    super(Rec, self).__init__()
    self.verbose = verbose
    rnn = layers.GRU if cfg.rec_type == 'gru' else layers.LSTM
    self.rec = rnn(units=cfg.rec_size, name='rec')
    self.fc = layers.Dense(num_classes, activation='softmax', name='fc')

  def call(self, x, training=False):

    if self.verbose:
      print(f'x {x.shape}')

    # (N, 16, 512) =>
    # (N, M)
    x = self.rec(x)
    if self.verbose:
      print(f'rec {x.shape}')

    # (N, M) =>
    # (N, C)
    x = self.fc(x)
    if self.verbose:
      print(f'fc {x.shape}')

    return x
