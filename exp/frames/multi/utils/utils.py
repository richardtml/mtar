""" utils.py
"""

from collections import namedtuple

import numpy as np
from torch import Tensor

def build_datasets_cfg(cfg, dss_names=('hmdb51', 'ucf101')):
  dss = []
  DS = namedtuple('DS', ('name', 'split'))
  for name in dss_names:
    enabled = getattr(cfg, name, False)
    if enabled:
      split = getattr(cfg, 'split', 1)
      dss.append(DS(name, split))
  return dss

def batches_to_numpy(batches):
  return [[z.numpy() if isinstance(z, Tensor) else z for z in batch]
    for batch in batches]
