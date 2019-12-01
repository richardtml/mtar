""" rec.py

Simple Rec model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers

from exp.reps.multi.models.shared import BaseAR


class Rec(BaseAR):

  def __init__(self, cfg, verbose=False):
    super(Rec, self).__init__(cfg, verbose)
    RNN = layers.GRU if cfg.model_rec_type == 'gru' else layers.LSTM
    ret_seqs = [True] * cfg.model_rec_layers
    ret_seqs[-1] = False
    self.recs = []
    for i, ret_seq in enumerate(ret_seqs):
      if cfg.model_rec_bi:
        rec = layers.Bidirectional(
            RNN(units=cfg.model_rec_size,
              return_sequences=ret_seq, name=f'rec{i}'),
            merge_mode=cfg.model_rec_bi_merge)
      else:
        rec = RNN(units=cfg.model_rec_size,
            return_sequences=ret_seq, name=f'rec{i}')
      self.recs.append(rec)
      setattr(self, f'rec{i}', rec)

  def call_shared(self, x, training, verbose):
    # (N, S, F) => (N, M)
    for i, rec in enumerate(self.recs):
      x = rec(x)
      if verbose: print(f'rec{i} {x.shape}')
    # (N, F)
    return x
