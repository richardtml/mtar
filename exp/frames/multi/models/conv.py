""" conv.py

Simple Conv model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers

from exp.reps.multi.models.shared import BaseAR


class Conv(BaseAR):

  def __init__(self, cfg, verbose=False):
    super(Conv, self).__init__(cfg, verbose)
    self.conv1d = layers.Conv1D(
        filters=cfg.model_conv1d_filters, kernel_size=3,
        padding='same', activation='relu',
        use_bias=(not cfg.model_bn_in),
        name='conv1d')
    self.dropout = layers.SpatialDropout1D(cfg.model_dropout, name='do1d')
    self.gap = layers.GlobalAveragePooling1D(name='gap')


  def call_shared(self, x, training, verbose):
    # (N, S, R) => (N, S, F)
    x = self.conv1d(x)
    if verbose: print(f'conv1d {x.shape}')
    # (N, S, F) => (N, S, F)
    x = self.dropout(x, training)
    if verbose: print(f'dropout {x.shape}')
    # (N, S, F) => (N, F)
    x = self.gap(x)
    if verbose: print(f'gap {x.shape}')
    # (N, F)
    return x
