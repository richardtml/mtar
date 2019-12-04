""" conv.py

Simple Conv model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers

from exp.reps.multi.models.shared import BaseAR


class Conv(BaseAR):

  def __init__(self, cfg, verbose=False):
    super(Conv, self).__init__(cfg, verbose)
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
    self.gap = layers.GlobalAveragePooling1D(name='gap')


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
    # (N, S', F) => (N, F)
    x = self.gap(x)
    if verbose: print(f'{self.gap.name} {x.shape}')
    # (N, F)
    return x
