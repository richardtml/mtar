""" ucf101.py

Simple test for datasets.
"""


import os

import fire
import numpy as np
import torch

import ucf101
from common import config


DATASETS_DIR = config.get('DATASETS_DIR')
torch.manual_seed(config.get('SEED'))


def test(ds='ucf101', zet='train', split=1, batch_size=1,
    num_batches=1, min_seq=16, max_seq=16, print_dropped=True):
  """Simple test function."""
  if ds == 'ucf101':
    ds_dir = os.path.join(DATASETS_DIR, 'ucf101')
    build_dataloader = ucf101.build_dataloader

  dl = build_dataloader(ds_dir, zet, split=split, batch_size=batch_size,
      min_seq=min_seq, max_seq=max_seq, print_dropped=print_dropped)
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
