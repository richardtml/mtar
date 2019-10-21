""" rec.py

Simple Recurrent model for Action Recognition.
"""


import tensorflow as tf
from tensorflow.keras import layers

from experiments.multi.models.shared import HMDB51UCF101


class Rec(HMDB51UCF101):

  def __init__(self, cfg, verbose=False):
    super(Rec, self).__init__(verbose)
    self.verbose = verbose
    rnn = layers.GRU if cfg.rec_type == 'gru' else layers.LSTM
    self.rec = rnn(units=cfg.rec_size)


  def call_shared(self, x, verbose):
    # (N, 16, 512)
    if verbose:
      print(f'x {x.shape}')

    # (N, 16, 512) =>
    # (N, M)
    x = self.rec(x)
    if verbose:
      print(f'rec {x.shape}')

    return x