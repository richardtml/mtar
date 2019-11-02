""" ucf101.py

Simple test for datasets.
"""


import os

import fire
import numpy as np
import torch

from builder import build_dataloader
from common import config


def test(ds, subset='train', split=1, batch_size=1, num_batches=1,
    min_seq=16, max_seq=16, verbose=True, print_dropped=True):
  """Simple test function."""
  datasets_dir = config.get('DATASETS_DIR')
  dl = build_dataloader(datasets_dir, ds, split=split, subset=subset,
      batch_size=batch_size, min_seq=min_seq, max_seq=max_seq,
      verbose=verbose, print_dropped=print_dropped)
  print('Traversing')
  for x, y in dl.take(num_batches):
    print(f'x.shape {x.shape}')
    print(f'x[0,0,:10]')
    print(x[0,0,:10])
    print(f'y.shape {y.shape}')
    print(f'y[0]')
    print(y[0])


if __name__ == '__main__':
  fire.Fire(test)
