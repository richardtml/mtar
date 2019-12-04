""" rec.py

Simple Recurrent model for Action Recognition.
"""


import tensorflow as tf
from tensorflow.keras import layers

from exp.reps.multi.models.shared import BaseAR


class Rec(BaseAR):

  def __init__(self, cfg, verbose=False):
    super(Rec, self).__init__(cfg, verbose)
    rnn = layers.GRU if cfg.model_rec_type == 'gru' else layers.LSTM
    self.rec = rnn(units=cfg.model_rec_size, name='rec')

  def call_shared(self, x, training, verbose):
    # (N, 16, 512) => (N, M)
    x = self.rec(x)
    if verbose: print(f'rec {x.shape}')
    # (N, M)
    return x
