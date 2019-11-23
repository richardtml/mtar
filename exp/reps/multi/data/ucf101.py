""" ucf101.py

UFC101 Dataset.
"""


import csv
from functools import lru_cache
from os.path import join

import numpy as np
import zarr
from torch.utils.data import Dataset

from common.data import DataLoader


class UCF101Dataset(Dataset):
  """UCF101 dataset."""

  def __init__(self, ds_dir, split, subset, min_seq=16, max_seq=16,
      sampling='equal', cache=False, verbose=False, print_dropped=False):
    if split not in (1, 2, 3):
      raise ValueError(f'invalid split={split}')
    if subset not in ('train', 'test'):
      raise ValueError(f'invalid subset={subset}')
    if max_seq < min_seq:
      raise ValueError(f"invalid min_seq={min_seq}<{max_seq}=max_seq")
    if sampling not in ('equal', 'random'):
      raise ValueError(f'invalid sampling={sampling}')
    self.max_seq = max_seq
    self.sampling = sampling
    cache = False if sampling == 'random' else cache
    zarr_dir = join(ds_dir, 'resnet50_0512.zarr')
    ds = zarr.open(zarr_dir, 'r')
    splits = zarr.open(join(ds_dir, 'splits.zarr'), 'r')
    names = list(splits[str(split)][subset][:])
    total = len(names)
    new_names, dropped = [], []
    for name in names:
      seq = ds[name]['x'].shape[0]
      if seq >= min_seq:
        new_names.append(name)
      else:
        dropped.append((name, seq))
    self.names = new_names
    self.ds = ds
    if cache:
      if verbose: print('Caching __getitem__ method')
      self.get_item = lru_cache(maxsize=len(self.names))(self.get_item)
    if verbose:
      print(
        f"Loading dataset {zarr_dir}\n"
        f" with subset={subset}"
        f" split={split}"
        f" samples={len(self.names)}"
        f" min_seq={min_seq}"
        f" total={total}"
        f" dropped={total - len(self.names)}"
      )
      if print_dropped:
        print('Dropped examples:')
        for d in dropped:
          print(f"  {d[0]} {d[1]}")

  def get_item(self, i):
    g = self.ds[self.names[i]]
    x = g['x']
    seq_size = x.shape[0]
    if self.sampling == 'equal':
      step = seq_size // self.max_seq
      idx = np.arange(0, seq_size, step)[:self.max_seq]
    else:
      idx = np.sort(np.random.choice(np.arange(seq_size), size=self.max_seq))
    x = np.array(x.get_orthogonal_selection(idx))
    y = np.array(g['y'])
    return x, y

  def __getitem__(self, i):
    return self.get_item(i)

  # def __getitem__(self, i):
  #   g = self.ds[self.names[i]]
  #   x = g['x']
  #   seq_size = x.shape[0]
  #   step = seq_size // self.max_seq
  #   idx = np.arange(0, seq_size, step)[:self.max_seq]
  #   x = np.array(x.get_orthogonal_selection(idx))
  #   y = np.array(g['y'])
  #   return x, y

  def __len__(self):
    return len(self.names)

