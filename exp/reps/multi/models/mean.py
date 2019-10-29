""" mean.py

Simple Mean model for Action Recognition.
"""


import tensorflow as tf
from tensorflow.keras import layers

from exp.reps.multi.models.shared import HMDB51UCF101


class MeanFC(HMDB51UCF101):

  def __init__(self, cfg, verbose=False):
    super(MeanFC, self).__init__(cfg.batchnorm, verbose)
    self.gap = layers.GlobalAveragePooling1D(name='gap')

  def call_shared(self, x, training, verbose):
    # (N, 16, R) => (N, R)
    x = self.gap(x)
    if verbose: print(f'gap {x.shape}')
    # (N, R)
    return x
