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
      num_frames=16, sampling='fixed', cache=False,
      verbose=False, print_dropped=False):

    if split not in (1, 2, 3):
      raise ValueError(f'invalid split={split}')
    if subset not in ('train', 'test'):
      raise ValueError(f'invalid subset={subset}')
    if num_frames < 1:
      raise ValueError(f"invalid num_frames={num_frames}")
    if sampling not in ('fixed', 'random'):
      raise ValueError(f'invalid sampling={sampling}')

    self.transform = transform
    self.num_frames = num_frames
    self.sampling = sampling
    self.cache = cache
    self.frames_dir = join(ds_dir, 'frames')
    self.ds = []

    splits = zarr.open(join(ds_dir, 'splits.zarr'), 'r')
    split_names = set(splits[str(split)][subset][:])
    dropped = []
    classes_names = sorted(listdir(self.frames_dir), key=str.casefold)
    for y, class_name in enumerate(classes_names):
      if class_name[0] != '.':
        class_path = join(self.frames_dir, class_name)
        for video_name in sorted(listdir(class_path), key=str.casefold):
          if video_name in split_names:
            subpath = join(class_name, video_name)
            video_num_frames = len(listdir(join(class_path, video_name)))
            if video_num_frames >= num_frames:
              self.ds.append([subpath, y])
            else:
              dropped.append([subpath, video_num_frames])

    if verbose:
      print(
        f"Loading dataset {ds_dir}\n"
        f"  with split={split}"
        f" subset={subset}"
        f" samples={len(self)}"
        f" num_frames={num_frames}"
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
    frames = []
    for frame_path in sorted(listdir(video_dir)):
      frame = Image.open(join(video_dir, frame_path))
      if self.cache:
        frame.load()
      frames.append(frame)
    return frames, y

  def __getitem__(self, i):
    frames, y = self.load_video(i)

    num_frames = len(frames)
    if self.sampling == 'fixed':
      step = num_frames // self.num_frames
      idx = np.arange(0, num_frames, step)[:self.num_frames]
    else:
      idx = np.arange(num_frames)
      idx = np.random.choice(idx, size=self.num_frames, replace=False)

    frames = [frame for i, frame in enumerate(frames) if i in idx]

    if self.transform:
      frames = self.transform(frames)
    x = np.stack([np.asarray(frame) for frame in frames])

    return x, y

  def __len__(self):
    return len(self.ds)
