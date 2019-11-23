""" conv2d.py

Simple Conv2D model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers

from exp.reps.multi.models.shared import BaseAR


class Conv2D(BaseAR):

  def __init__(self, cfg, verbose=False):
    super(Conv2D, self).__init__(cfg, verbose)
    self.pad = layers.ZeroPadding2D(padding=(1, 0), name='zpad2d')
    self.conv2d = layers.Conv2D(
        filters=cfg.model_conv2d_filters, kernel_size=(3, 512),
        padding='valid', activation='relu',
        use_bias=(not cfg.model_bn_in),
        name='conv2d')
    self.dropout = layers.SpatialDropout1D(cfg.model_dropout, name='do1d')
    self.gap = layers.GlobalAveragePooling1D(name='gap')

  def call_shared(self, x, training, verbose):
    # (N, 16, R) => (N, 16, R, 1)
    x = tf.expand_dims(x, 3)
    if verbose: print(f'expand_dims {x.shape}')
    # (N, 16, R, 1) => (N, 18, R, 1)
    x = self.pad(x)
    if verbose: print(f'pad {x.shape}')
    # (N, 18, R, 1) => (N, 16, 1, F)
    x = self.conv2d(x)
    if verbose: print(f'conv2d {x.shape}')
    # (N, 16, 1, F) => (N, 16, F)
    x = tf.squeeze(x, 2)
    if verbose: print(f'squeeze {x.shape}')
    # (N, 16, F) => (N, 16, F)
    x = self.dropout(x, training)
    if verbose: print(f'dropout {x.shape}')
    # (N, 16, F) => (N, F)
    x = self.gap(x)
    if verbose: print(f'gap {x.shape}')
    # (N, F)
    return x
