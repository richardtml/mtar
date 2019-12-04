""" ards.py

Action recognition dataset to load reps.
"""

import os
from os import listdir
from os.path import join

import numpy as np
import zarr
from torch.utils.data import Dataset


def select_indices(num_frames, max_frames, sampling):
  if sampling == 'fixed':
    step = num_frames // max_frames
    ids = np.arange(0, num_frames, step)[:max_frames]
    offset = ((num_frames - 1) - ids[-1]) // 2
    ids += offset
  else: # random
    ids = np.arange(num_frames)
    ids = np.sort(np.random.choice(ids, size=max_frames))
  return ids

class ARFramesDS(Dataset):
  """Action recognition dataset to load frames."""

  def __init__(self, ds_dir, split, subset,
      num_frames=16, sampling='fixed', cache=False,
      verbose=False, print_dropped=False):

    if split not in {1, 2, 3}:
      raise ValueError(f'invalid split={split}')
    if subset not in {'train', 'test'}:
      raise ValueError(f'invalid subset={subset}')
    if num_frames < 1:
      raise ValueError(f"invalid num_frames={num_frames}")
    if sampling not in {'fixed', 'random'}:
      raise ValueError(f'invalid sampling={sampling}')

    self.num_frames = num_frames
    self.sampling = sampling
    self.cache = cache
    self.frames_dir = join(ds_dir, 'frames')
    self.ds = []

    self.ds = zarr.open(join(ds_dir, 'resnet50_0512.zarr'), 'r')
    splits = zarr.open(join(ds_dir, 'splits.zarr'), 'r')
    split_names = set(splits[str(split)][subset][:])
    self.names = []
    dropped = []
    for name in list(splits[str(split)][subset][:]):
      video_num_frames = self.ds[name]['x'].shape[0]
      if video_num_frames >= num_frames:
        self.names.append(name)
      else:
        dropped.append([name, video_num_frames])

    if verbose:
      print(
        f"Loading dataset {ds_dir}\n"
        f"  with split={split}"
        f" subset={subset}"
        f" samples={len(self)}"
        f" num_frames={num_frames}"
        f" total={len(split_names)}"
        f" dropped={len(dropped)}"
      )
    if self.cache:
      if verbose: print(f"Enabling cache")
      self.frames_cache = {}
    if print_dropped:
      print('Dropped examples:')
      for subpath, num_frames in dropped:
        print(f"  {subpath} {num_frames}")

  def get_cached_item(self, i):
    example = self.frames_cache.get(i, None)
    if example is None:
      g = self.ds[self.names[i]]
      x, y = g['x'], g['y']
      x = np.array(x)
      y = np.array(y)
      example = (x, y)
      self.frames_cache[i] = example
    x, y = example
    ids = select_indices(x.shape[0], self.num_frames, self.sampling)
    x = x[ids]
    return x, y

  def get_item(self, i):
    g = self.ds[self.names[i]]
    x, y = g['x'], g['y']
    ids = select_indices(x.shape[0], self.num_frames, self.sampling)
    x = np.array(x.get_orthogonal_selection(ids))
    y = np.array(y)
    return x, y

  def __getitem__(self, i):
    return self.get_cached_item(i) if self.cache else self.get_item(i)

  def __len__(self):
    return len(self.names)
