""" ucf101.py

UFC101 Dataset.
"""


import csv
import glob
from os.path import join

import numpy as np
import zarr
from torch.utils.data import Dataset

from common.data import DataLoader


class HMDB51Dataset(Dataset):
  """HMDB51 dataset."""

  def __init__(self, ds_dir, subset, split=1,
      min_seq=16, max_seq=16, print_dropped=False):
    if split not in (1, 2, 3):
      raise ValueError(f'invalid split={split}')
    if subset not in ('train', 'test'):
      raise ValueError(f'invalid subset={subset}')
    if max_seq < min_seq:
      raise ValueError(f"invalid min_seq={min_seq}<{max_seq}=max_seq")
    self.max_seq = max_seq
    zarr_dir = join(ds_dir, 'hmdb51_resnet50_0512.zarr')
    self.ds = zarr.open(zarr_dir, 'r')
    self.names = load_split(ds_dir, subset, split)
    total = len(self.names)
    new_names, dropped = [], []
    for name in self.names:
      seq = self.ds[name]['x'].shape[0]
      if seq >= min_seq:
        new_names.append(name)
      else:
        dropped.append((name, seq))
    self.names = new_names
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

  def __getitem__(self, i):
    g = self.ds[self.names[i]]
    x = g['x']
    seq_size = x.shape[0]
    step = seq_size // self.max_seq
    idx = np.arange(0, seq_size, step)[:self.max_seq]
    x = np.array(x.get_orthogonal_selection(idx))
    y = np.array(g['y'])
    return x, y

  def __len__(self):
    return len(self.names)


def load_split(ds_dir, subset, split):
  """Loads examples names in split."""
  pattern = join(ds_dir, 'splits', f'*split{split}.txt')
  names = []
  for filepath in sorted(glob.glob(pattern)):
    names.extend(load_file_class(filepath, subset))
  return names

def load_file_class(path, subset):
  subset = 1 if subset == 'train' else 2
  names = []
  with open(path, 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
      if subset == int(row[1]):
        name = row[0].split('.')[0]
        names.append(name)
  return names