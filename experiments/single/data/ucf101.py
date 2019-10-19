""" ucf101.py

UFC101 Dataset.
"""


import csv
from os.path import join

import numpy as np
import zarr
from torch.utils.data import Dataset

from common.data import DataLoader


class UCF101Dataset(Dataset):
  """UCF101 dataset."""

  def __init__(self, ds_dir, set_name, split=1,
      min_seq=16, max_seq=16, print_dropped=False):
    if split not in (1, 2, 3):
      raise ValueError(f'invalid split={split}')
    if set_name not in ('train', 'test'):
      raise ValueError(f'invalid set_name="{set_name}"')
    if max_seq < min_seq:
      raise ValueError(f"invalid min_seq={min_seq}<{max_seq}=max_seq")
    self.max_seq = max_seq
    zarr_dir = join(ds_dir, 'ucf101_resnet50_0512.zarr')
    self.ds = zarr.open(zarr_dir, 'r')
    split_path = join(ds_dir, 'splits', f'{set_name}list0{split}.txt')
    self.names = load_split(split_path)
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
      f"using split {split_path}\n"
      f"with {len(self.names)} examples "
      f"with min_seq={min_seq} out of {total}, "
      f"dropped {total - len(self.names)}"
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


def load_split(path):
  """Loads examples names in split."""
  names = []
  with open(path, 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
      name = row[0].split('/')[1].split('.')[0]
      names.append(name)
  return names


def build_dataloader(ds_dir, set_name, split, batch_size,
    min_seq=16, max_seq=16, num_workers=4, print_dropped=False):
  """Loads examples names in split."""
  ds = UCF101Dataset(ds_dir, set_name=set_name, split=split,
      min_seq=min_seq, max_seq=max_seq, print_dropped=print_dropped)
  dl = DataLoader(ds, batch_size=batch_size,
      shuffle=True, num_workers=num_workers)
  return dl
