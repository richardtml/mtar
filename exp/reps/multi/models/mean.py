""" mean.py

Simple Mean model for Action Recognition.
"""

from collections import namedtuple

import tensorflow as tf
from tensorflow.keras import layers

from exp.reps.multi.models.shared import BaseAR


FC_CLASSES = {
  'hmdb51': 51,
  'ucf101': 101,
}


class MeanFC(BaseAR):

  def __init__(self, cfg, verbose=False):
    super(MeanFC, self).__init__(cfg, verbose)
    self.gap = layers.GlobalAveragePooling1D(name='gap')

  def call_shared(self, x, training, verbose):
    # (N, 16, R) => (N, R)
    x = self.gap(x)
    if verbose: print(f'gap {x.shape}')
    # (N, R)
    return x


class FCMean(tf.keras.Model):

  def __init__(self, cfg, verbose):
    super(FCMean, self).__init__()
    self.verbose = verbose
    self.bn_in = None
    if cfg.model_bn_in:
      name = f'bn_in'
      bn = layers.BatchNormalization(name=name)
      setattr(self, name, bn)
      self.bn_in = bn
    Task = namedtuple('Task', ('name', 'bn_out', 'fc'))
    self.tasks = []
    for ds in cfg._dss:
      bn_out = None
      if cfg.model_bn_out:
        name = f'{ds.name}_bn_out'
        bn_out = layers.BatchNormalization(name=name)
        setattr(self, name, bn_out)
      name = f'{ds.name}_fc'
      size = FC_CLASSES[ds.name]
      fc = layers.Dense(size, activation='softmax', name=name)
      setattr(self, name, fc)
      self.tasks.append(Task(ds.name, bn_out, fc))
    self.gap = layers.GlobalAveragePooling1D(name='gap')

  def call_in_shared(self, x, training, verbose):
    # (N, 16, R)
    if verbose: print(f'x {x.shape}')
    if self.bn_in:
      # (N, 16, R) => (N, 16, R)
      x = self.bn_in(x, training)
      if verbose: print(f'bn_in {x.shape}')
    return x

  def call(self, xs, training=False):
    # [T, (N, 16, R)]
    if len(xs) != len(self.tasks):
      raise ValueError(f'len(xs)=={len(xs)}!={len(self.tasks)}==len(tasks)')
    xs = list(xs)
    verbose = self.verbose
    if verbose: print(f'xs {[tuple(x.shape) for x in xs]}')
    verbose_shared = verbose
    for i, (x, task) in enumerate(zip(xs, self.tasks)):
      if x is not None:
        # (N, 16, R) => (N, 16, R)
        x = self.call_in_shared(x, training, verbose_shared)
        shape = x.shape
        # (N, 16, R) => (N*16, R)
        x = tf.reshape(x, (-1, shape[2]))
        if verbose: print(f'reshape {x.shape}')
        # (N*16, R) => (N*16, C)
        x = task.fc(x)
        if verbose: print(f'fc {x.shape}')
        # (N*16, C) => (N, 16, C)
        x = tf.reshape(x, (shape[0], shape[1], -1))
        if verbose: print(f'reshape {x.shape}')
        # (N, 16, C) => (N, C)
        x = self.gap(x)
        if verbose: print(f'gap {x.shape}')
        xs[i] = x
        if verbose_shared:
          verbose_shared = False
    # [T, (N, C)]
    if verbose: print(f'output {[tuple(x.shape) for x in xs]}')
    return xs
