""" builder.py

Datasets Builder.
"""

from os.path import join

import numpy as np
from torch.utils.data import DataLoader

from exp.reps.multi.data.ards import ARFramesDS


def build_dataloader(datasets_dir, ds, split, subset,
    num_frames=16, sampling='fixed', cache=False,
    batch_size=16, shuffle=True, num_workers=2,
    verbose=False, print_dropped=False):
  """Builds a Dataloader."""

  if ds not in {'hmdb51', 'ucf101'}:
      raise ValueError(f'invalid split={split}')
  if verbose:
    print(f'Building dataloader for {ds}')

  ds_dir = join(datasets_dir, ds)

  ds = ARFramesDS(ds_dir,
      split=split, subset=subset,
      num_frames=num_frames, sampling=sampling, cache=cache,
      verbose=verbose, print_dropped=print_dropped)
  dl = DataLoader(ds, batch_size=batch_size,
      shuffle=shuffle, num_workers=num_workers, collate_fn=collate)

  return dl

def collate(batch):
  """Custom collate to avoid producing torch.Tensor."""
  xs, ys = zip(*batch)
  return np.stack(xs), np.array(ys, dtype=np.int32)
