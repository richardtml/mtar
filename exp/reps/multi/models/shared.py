""" rec.py

Shared base model for Action Recognition.
"""

from collections import namedtuple

import tensorflow as tf
from tensorflow.keras import layers



FC_CLASSES = {
  'hmdb51': 51,
  'ucf101': 101,
}


class BaseAR(tf.keras.Model):

  def __init__(self, cfg, verbose):
    super(BaseAR, self).__init__()
    self.verbose = verbose
    self.cfg = cfg
    self.bn_in = None
    if cfg.model.bn_in:
      name = f'bn_in'
      bn = layers.BatchNormalization(name=name)
      setattr(self, name, bn)
      self.bn_in = bn

    Task = namedtuple('Task', ('name', 'bn_out', 'fc'))
    self.tasks = []
    for task_name, task_prop in cfg.tasks.items():
      bn_out = None
      if task_prop.get('bn_out', None):
        name = f'{task_name}_bn_out'
        bn_out = layers.BatchNormalization(name=name)
        # self.bns_out.append(bn_out)
        setattr(self, name, bn_out)
      name = f'{task_name}_fc'
      size = FC_CLASSES[task_name]
      fc = layers.Dense(size, activation='softmax', name=name)
      # self.fcs.append(layer)
      setattr(self, name, fc)
      self.tasks.append(Task(task_name, bn_out, fc))
      # self.tasks[task_name] = {'bn_out': bn_out, 'fc': fc}

  def call_in_shared(self, x, training, verbose):
    # (N, 16, R)
    if verbose: print(f'x {x.shape}')
    if self.bn_in:
      # (N, 16, R) => (N, 16, R)
      x = self.bn_in(x, training)
      if verbose: print(f'bn_in {x.shape}')
    return x

  def call_shared(self, x, training, verbose):
    return x

  def call(self, xs, training=False):
    # [T, (N, 16, R)]
    # if isinstance(xs, (np.ndarray, tf.Tensor)):
    #   xs = tf.unstack(xs)
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
        # (N, 16, R) => (N, F)
        x = self.call_shared(x, training, verbose_shared)
        if task.bn_out is not None:
          # (N, F) => (N, F)
          x = task.bn_out(x, training)
          if verbose: print(f'{task.name}_bn_out {x.shape}')
        # (N, C) => (N, C)
        x = task.fc(x)
        if verbose: print(f'{task.name}_fc {x.shape}')
        xs[i] = x
        if verbose_shared:
          verbose_shared = False
    # [T, (N, C)]
    if verbose: print(f'output {[tuple(x.shape) for x in xs]}')
    return xs
