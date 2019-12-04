""" conv.py

Simple Conv model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers

from exp.reps.multi.models.shared import BaseAR


class ConvRec(BaseAR):

  def __init__(self, cfg, verbose=False):
    super(ConvRec, self).__init__(cfg, verbose)
    self.convs = []
    for i, filters in enumerate(cfg.model_conv_filters):
      use_bias = not cfg.model_bn_in if i == 0 else True
      conv = layers.Conv1D(
        filters=filters, kernel_size=3, padding='same',
        activation='relu', use_bias=use_bias, name=f'conv{i}')
      self.convs.append(conv)
      setattr(self, conv.name, conv)
    self.pool = layers.MaxPool1D(pool_size=2, name='pool')
    self.dropout = layers.SpatialDropout1D(cfg.model_dropout, name='dropout')
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
    # (N, S, R) => (N, S', F)
    for conv in self.convs:
      # (N, S, R) => (N, S, F)
      x = conv(x)
      if verbose: print(f'{conv.name} {x.shape}')
      # (N, S, F) => (N, S/2, F)
      x = self.pool(x)
      if verbose: print(f'{self.pool.name} {x.shape}')
      # (N, S/2, F)
      x = self.dropout(x, training)
      if verbose: print(f'{self.dropout.name} {x.shape}')
    # (N, S, F) => (N, M)
    for rec in self.recs:
      x = rec(x)
      if verbose: print(f'{rec.name} {x.shape}')
    # (N, F)
    return x
