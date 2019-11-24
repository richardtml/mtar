""" builder.py

Datasets Builder.
"""

from os.path import join

import torch

from ards import ARFramesDS
from vt import VideoTransform
from common.data import DataLoader


def build_dataloader(datasets_dir,
    ds, split, subset, batch_size, sampling='fixed',
    min_seq=16, max_seq=16, shuffle=True, num_workers=2,
    cache=False, verbose=False, print_dropped=False):
  """Builds a Dataloader."""

  if ds not in ('hmdb51', 'ucf101'):
      raise ValueError(f'invalid split={split}')
  if verbose:
    print(f'Building dataloader for {ds}')

  print(':)')
  ds_dir = join(datasets_dir, ds)
  transform = VideoTransform()
  ds = ARFramesDS(ds_dir, split=split, subset=subset, transform=transform,
      min_seq=min_seq, max_seq=max_seq, cache=cache, sampling=sampling,
      verbose=verbose, print_dropped=print_dropped)
  dl = DataLoader(ds, batch_size=batch_size,
      shuffle=shuffle, num_workers=num_workers)

  return dl
