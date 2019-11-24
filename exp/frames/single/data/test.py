""" test.py

Simple test for datasets.
"""

import math
import os

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch

from builder import build_dataloader
from common import config

def plot_clip(clip):
  height = math.ceil(math.sqrt(len(clip)))
  width = height if height * height == len(clip) else height + 1
  fig = plt.figure(figsize=(height*2, width*2))
  for i, frame in enumerate(clip):
    ax = fig.add_subplot(width, height, i+1)
    ax.imshow(frame)
    ax.set_xticks([])
    ax.set_yticks([])
  plt.tight_layout()
  plt.show()

def test(ds, subset='train', split=1, batch_size=1, batches=1, epochs=1,
    sampling='fixed', cache=False, min_seq=16, max_seq=16,
    shuffle=False, num_workers=0, verbose=True, print_dropped=True, plot=False):
  """Simple test function."""
  datasets_dir = config.get('DATASETS_DIR')
  dl = build_dataloader(datasets_dir, ds, split=split, subset=subset,
      batch_size=batch_size, sampling=sampling, shuffle=shuffle,
      min_seq=min_seq, max_seq=max_seq,
      num_workers=num_workers, verbose=verbose, print_dropped=print_dropped)

  print(f'Number of batches {len(dl)}')
  for epoch in range(epochs):
    print(f'Epoch {epoch}')
    for x, y in dl.take(batches):
      print(
        f"  x.shape {x.shape}\n"
        f"  x.flatten()[:10] {x.flatten()[:10]}\n"
        f"  y.shape {y.shape}\n"
        f"  y[0] {y[0]}"
      )
      if plot:
        clip = [x[0][i] for i in range(len(x[0]))]
        plot_clip(clip)


if __name__ == '__main__':
  fire.Fire(test)
