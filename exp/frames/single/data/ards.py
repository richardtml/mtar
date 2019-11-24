""" ards.py

Action recognition dataset to load frames.
"""

import csv
import glob
from functools import lru_cache
from os import listdir
from os.path import join

import numpy as np
import zarr
from PIL import Image
from torch.utils.data import Dataset

from common.data import DataLoader

class ARFramesDS(Dataset):
  """Action recognition dataset to load frames."""

  def __init__(self, ds_dir, split, subset, transform=None,
      min_seq=16, max_seq=16, sampling='fixed', cache=False,
      verbose=False, print_dropped=False):

    if split not in (1, 2, 3):
      raise ValueError(f'invalid split={split}')
    if subset not in ('train', 'test'):
      raise ValueError(f'invalid subset={subset}')
    if max_seq < min_seq:
      raise ValueError(f"invalid min_seq={min_seq}<{max_seq}=max_seq")
    if sampling not in ('fixed', 'random'):
      raise ValueError(f'invalid sampling={sampling}')

    self.transform = transform
    self.max_seq = max_seq
    self.sampling = sampling
    self.frames_dir = join(ds_dir, 'frames')
    self.ds = []

    splits = zarr.open(join(ds_dir, 'splits.zarr'), 'r')
    split_names = set(splits[str(split)][subset][:])
    dropped = []
    for y, class_name in enumerate(sorted(listdir(self.frames_dir))):
      class_path = join(self.frames_dir, class_name)
      for video_name in sorted(listdir(class_path)):
        if video_name in split_names:
          subpath = join(class_name, video_name)
          num_frames = len(listdir(join(class_path, video_name)))
          if num_frames >= min_seq:
            self.ds.append([subpath, y])
          else:
            dropped.append([subpath, num_frames])

    if verbose:
      print(
        f"Loading dataset {ds_dir}\n"
        f"  with split={split}"
        f" subset={subset}"
        f" samples={len(self)}"
        f" min_seq={min_seq}"
        f" max_seq={max_seq}"
        f" total={len(split_names)}"
        f" dropped={len(split_names) - len(self)}"
      )
    if cache:
      if verbose: print(f"Enabling cache")
      self.load_video = lru_cache(maxsize=len(self))(self.load_video_)
    else:
      self.load_video = self.load_video_
    if print_dropped:
      print('Dropped examples:')
      for subpath, num_frames in dropped:
        print(f"  {subpath} {num_frames}")

  def load_video_(self, i):
    subpath, y = self.ds[i]
    video_dir = join(self.frames_dir, subpath)
    frames = [Image.open(join(video_dir, frame_path))
        for frame_path in sorted(listdir(video_dir))]
    return frames, y

  def __getitem__(self, i):
    frames, y = self.load_video(i)

    seq_size = len(frames)
    if self.sampling == 'fixed':
      step = seq_size // self.max_seq
      idx = np.arange(0, seq_size, step)[:self.max_seq]
    else:
      print(f'seq_size {seq_size}')
      idx = np.arange(seq_size)
      idx = np.random.choice(idx, size=self.max_seq, replace=False)

    frames = [frame for i, frame in enumerate(frames) if i in idx]

    if self.transform:
      frames = self.transform(frames)
    x = np.stack([np.asarray(frame) for frame in frames])

    return x, y

  def __len__(self):
    return len(self.ds)
