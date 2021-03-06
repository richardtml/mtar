""" ucf101.py

Simple test for datasets.
"""

import os
from itertools import islice

import fire
import numpy as np
import torch

from builder import build_dataloader
from common import config

torch.manual_seed(config.get('SEED'))

def test(ds, subset='train', split=1, batch_size=1, batches=1, epochs=1,
    num_frames=16, shuffle=False, sampling='fixed', cache=False,
    verbose=True, print_dropped=True):
  """Simple test function."""
  datasets_dir = config.get('DATASETS_DIR')
  dl = build_dataloader(datasets_dir, ds, split=split, subset=subset,
      batch_size=batch_size, num_frames=num_frames,
      shuffle=shuffle, sampling=sampling, cache=cache,
      verbose=verbose, print_dropped=print_dropped)
  print('Traversing')
  print(f'Number of batches {len(dl)}')
  for epoch in range(epochs):
    print(f'Epoch {epoch}')
    for batch, (x, y) in enumerate(islice(dl, batches)):
      print(f'batch {batch}')
      print(f'x.shape {x.shape}')
      print(f'x[0,0,:10]')
      print(x[0,0,:10])
      print(f'y.shape {y.shape}')
      print(f'y[0]')
      print(y[0])

if __name__ == '__main__':
  fire.Fire(test)
