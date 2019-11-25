""" utils.py
"""

from functools import partial
from itertools import zip_longest

import numpy as np

def tzip_refill(*dls):
  sizes = [len(dl) for dl in dls]
  i_max_dl = np.argmax(sizes)
  its = [iter(dl) for dl in dls]
  while True:
    batches = []
    for i in range(len(its)):
      try:
        batch = next(its[i])
        batches.append(batch)
      except StopIteration:
        if i != i_max_dl:
          its[i] = iter(dls[i])
          batch = next(its[i])
          batches.append(batch)
        else:
          return
    yield batches

def tzip_interleave(*dls, fillvalue=(None, None)):
  sizes = np.array([len(dl) for dl in dls])
  max_size = np.max(sizes)
  steps = [max_size // size for size in sizes]
  masks = [np.zeros(max_size) for _ in sizes]
  for mask, size, step in zip(masks, sizes, steps):
    indices = np.arange(0, size*step, step)
    remaining = (max_size - 1) - indices[-1]
    if remaining:
      #print('indices', indices)
      #print('remaining', remaining)
      offset = np.arange(remaining)
      #print('offset', offset)
      if max_size % size:
        indices[-len(offset):] = indices[-len(offset):] + offset[:len(indices)]
    np.put(mask, indices, np.ones_like(indices))
  its = [iter(dl) for dl in dls]
  for batches_mask in zip(*masks):
    batches = []
    for batch_mask, it in zip(batches_mask, its):
      if batch_mask:
        batch = next(it)
      else:
        batch = fillvalue
      batches.append(batch)
    yield batches

class TZipShortest:

  def __init__(self, *dls):
    self.len = min(len(dl) for dl in dls)
    self.its = [iter(dl) for dl in dls]

  def __iter__(self):
    return self

  def __len__(self):
    return self.len

  def __next__(self):
    return [next(it) for it in self.its]


def build_tzip(cfg):
  """Builds training strategy zip."""
  strategy = cfg.train_strategy
  if strategy == 'shortest':
    # return zip
    return TZipShortest
  elif strategy == 'longest':
    return partial(zip_longest, fillvalue=(None, None))
  elif strategy == 'refill':
    return tzip_refill
  elif strategy == 'interleave':
    return tzip_interleave
  else:
    raise ValueError(f'invalid param train_strategy={strategy}')
