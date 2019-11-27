""" ards.py

Action recognition dataset to load frames.
"""

import io
import os
from os import listdir
from os.path import join

import numpy as np
import zarr
from PIL import Image
from torch.utils.data import Dataset

def load_binary_image(path):
  with open(path, 'rb') as f:
    return bytearray(f.read())

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

  def __init__(self, ds_dir, split, subset, transform,
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
    classes_names = [c for c in classes_names if c[0] != '.']
    for y, class_name in enumerate(classes_names):
      class_dir = join(self.frames_dir, class_name)
      for video_name in sorted(listdir(class_dir), key=str.casefold):
        if video_name in split_names:
          subpath = join(class_name, video_name)
          video_num_frames = len(listdir(join(class_dir, video_name)))
          if video_num_frames >= num_frames:
            self.ds.append([subpath, video_num_frames, y])
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
    if self.cache:
      if verbose: print(f"Enabling cache")
      self.frames_cache = {}
    if print_dropped:
      print('Dropped examples:')
      for subpath, num_frames in dropped:
        print(f"  {subpath} {num_frames}")

  def load_binary_image(self, vf, path):
    if self.cache:
      bytez = self.frames_cache.get(vf, None)
      if bytez is None:
        bytez = load_binary_image(path)
        self.frames_cache[vf] = bytez
    else:
      bytez = load_binary_image(path)
    return bytez

  def load_frame(self, vf, video_dir):
    frame_path = join(video_dir, f'{vf[1]:03d}.jpg')
    bytez = self.load_binary_image(vf, frame_path)
    frame = Image.open(io.BytesIO(bytez))
    return frame

  def load_clip(self, v, subdir, ids):
    video_dir = join(self.frames_dir, subdir)
    return [self.load_frame((v, f), video_dir) for f in ids]

  def __getitem__(self, i):
    subdir, num_frames, y = self.ds[i]
    ids = select_indices(num_frames, self.num_frames, self.sampling)
    frames = self.load_clip(i, subdir, ids)
    x = self.transform(frames)
    return x, y

  def __len__(self):
    return len(self.ds)
