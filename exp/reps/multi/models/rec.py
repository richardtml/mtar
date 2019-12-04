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
    self.recs = []
    last_layer_idx = len(cfg.model_rec_sizes) - 1
    for i, rec_size in enumerate(cfg.model_rec_sizes):
      ret_seqs = i != last_layer_idx
      rec = RNN(units=rec_size, return_sequences=ret_seqs, name=f'rec{i}')
      if cfg.model_rec_bi:
        rec = layers.Bidirectional(rec,
          merge_mode=cfg.model_rec_bi_merge, name=f'rec_bi{i}')
      self.recs.append(rec)
      setattr(self, rec.name, rec)

  def call_shared(self, x, training, verbose):
    # (N, S, F) => (N, M)
    for rec in self.recs:
      x = rec(x)
      if verbose: print(f'{rec.name} {x.shape}')
    # (N, F)
    return x
