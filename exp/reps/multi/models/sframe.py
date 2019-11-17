""" mean.py

Simple single frame model for Action Recognition.
"""

import tensorflow as tf
from tensorflow.keras import layers

from exp.reps.multi.models.shared import BaseAR


class SFrame(BaseAR):

  def __init__(self, cfg, verbose=False):
    super(SFrame, self).__init__(cfg, verbose)

  def call_shared(self, x, training, verbose):
    # (N, 16, R) => (N, R)
    x = x[:, x.shape[1]//2]
    if verbose: print(f'middle {x.shape}')
    # (N, R)
    return x
