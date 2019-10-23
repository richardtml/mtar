""" builder.py

Datasets Builder.
"""

from os.path import join

import torch

from common.data import DataLoader
from common import config
from exp.reps.single.data.hmdb51 import HMDB51Dataset
from exp.reps.single.data.ucf101 import UCF101Dataset


torch.manual_seed(config.get('SEED'))


def build_dataloader(datasets_dir, ds, subset, split, batch_size,
    min_seq=16, max_seq=16, num_workers=4, print_dropped=False):
  """Builds a Dataloader."""
  if ds == 'hmdb51':
    Dataset = HMDB51Dataset
  elif ds == 'ucf101':
    Dataset = UCF101Dataset
  else:
    raise ValueError(f'invalid ds={ds}')
  ds_dir = join(datasets_dir, ds)
  ds = Dataset(ds_dir, subset=subset, split=split,
      min_seq=min_seq, max_seq=max_seq, print_dropped=print_dropped)
  dl = DataLoader(ds, batch_size=batch_size,
      shuffle=True, num_workers=num_workers)
  return dl
