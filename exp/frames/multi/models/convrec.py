""" convrec.py

Simple ConvRec model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers

from exp.reps.multi.models.shared import BaseAR


class ConvRec(BaseAR):

  def __init__(self, cfg, verbose=False):
    super(ConvRec, self).__init__(cfg, verbose)
    self.conv1d = layers.Conv1D(
        filters=cfg.model_conv1d_filters, kernel_size=3,
        padding='same', activation='relu',
        use_bias=(not cfg.model_bn_in),
        name='conv1d')
    self.dropout = layers.SpatialDropout1D(cfg.model_dropout, name='do1d')
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
    # (N, S, R) => (N, S, F)
    x = self.conv1d(x)
    if verbose: print(f'conv1d {x.shape}')
    # (N, S, F) => (N, S, F)
    x = self.dropout(x, training)
    if verbose: print(f'dropout {x.shape}')
    # (N, S, F) => (N, M)
    for i, rec in enumerate(self.recs):
      x = rec(x)
      if verbose: print(f'rec{i} {x.shape}')
    # (N, F)
    return x
