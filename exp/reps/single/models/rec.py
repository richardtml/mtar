""" rec.py

Simple Recurrent model for Action Recognition.
"""


import tensorflow as tf
from tensorflow.keras import layers


class Rec(tf.keras.Model):

  def __init__(self, cfg, num_classes, verbose=False):
    super(Rec, self).__init__()
    self.num_layers = cfg.rec_layers
    self.verbose = verbose
    RNN = layers.GRU if cfg.rec_type == 'gru' else layers.LSTM
    ret_seqs = [True] * self.num_layers
    ret_seqs[-1] = False
    for i, ret_seq in enumerate(ret_seqs):
      rnn = RNN(units=cfg.rec_size, return_sequences=ret_seq, name=f'rec{i}')
      setattr(self, f'rec{i}', rnn)
    self.fc = layers.Dense(num_classes, activation='softmax', name='fc')

  def call(self, x, training=False):
    verbose = self.verbose
    # (N, 16, R)
    if verbose: print(f'x {x.shape}')
    # (N, 16, R) => (N, M)
    for i in range(self.num_layers):
      rec = getattr(self, f'rec{i}')
      x = rec(x)
      if self.verbose: print(f'rec{i} {x.shape}')
    # (N, M) => (N, C)
    x = self.fc(x)
    if self.verbose: print(f'fc {x.shape}')
    # (N, C)
    return x
