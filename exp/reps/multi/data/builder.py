""" builder.py

Datasets Builder.
"""

from os.path import join

import torch

from common.data import DataLoader
from common import config
from exp.reps.multi.data.hmdb51 import HMDB51Dataset
from exp.reps.multi.data.ucf101 import UCF101Dataset


def build_dataloader(datasets_dir, ds, split, subset, batch_size,
    min_seq=16, max_seq=16, shuffle=True, num_workers=4,
    verbose=False, print_dropped=False):
  """Builds a Dataloader."""
  if ds == 'hmdb51':
    Dataset = HMDB51Dataset
  elif ds == 'ucf101':
    Dataset = UCF101Dataset
  else:
    raise ValueError(f'invalid ds={ds}')
  ds_dir = join(datasets_dir, ds)
  if verbose:
    print(f'Building dataloader for {ds}')
  ds = Dataset(ds_dir, split=split, subset=subset, min_seq=min_seq,
      max_seq=max_seq, verbose=verbose, print_dropped=print_dropped)
  dl = DataLoader(ds, batch_size=batch_size,
      shuffle=shuffle, num_workers=num_workers)
  return dl
