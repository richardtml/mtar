""" builder.py

Datasets Builder.
"""

from os.path import join

import torch

from exp.frames.multi.data.ards import ARFramesDS
from exp.frames.multi.data.vt import VideoShapeTransform, VideoTransform
# from common.data import DataLoader
from torch.utils.data import DataLoader


def build_dataloader(datasets_dir, ds, split, subset,
    transform=False, num_frames=16, sampling='fixed', cache=False,
    batch_size=16, shuffle=True, num_workers=2,
    verbose=False, print_dropped=False):
  """Builds a Dataloader."""

  if ds not in ('hmdb51', 'ucf101'):
      raise ValueError(f'invalid split={split}')
  if verbose:
    print(f'Building dataloader for {ds}')

  ds_dir = join(datasets_dir, ds)
  transform = VideoTransform() if transform else VideoShapeTransform()
  ds = ARFramesDS(ds_dir,
      split=split, subset=subset, transform=transform,
      num_frames=num_frames, sampling=sampling, cache=cache,
      verbose=verbose, print_dropped=print_dropped)
  dl = DataLoader(ds, batch_size=batch_size,
      shuffle=shuffle, num_workers=num_workers)

  return dl
