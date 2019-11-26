""" test.py

Simple test for datasets.
"""

import math
import os
from itertools import islice

import fire
import matplotlib.pyplot as plt

from builder import build_dataloader
from common import config

def plot_clip(clip):
  height = math.ceil(math.sqrt(len(clip)))
  width = height if height * height == len(clip) else height + 1
  fig = plt.figure(figsize=(width*2, height*2))
  for i, frame in enumerate(clip):
    ax = fig.add_subplot(width, height, i+1)
    ax.imshow(frame)
    ax.set_xticks([])
    ax.set_yticks([])
  plt.tight_layout()
  plt.show()

def test(ds, split=1, subset='train', transform=False,
    num_frames=16, sampling='fixed', cache=False,
    batch_size=1, shuffle=False, num_workers=0,
    batches=1, epochs=1,
    verbose=True, print_dropped=True, plot=False):
  """Simple test function."""

  datasets_dir = config.get('DATASETS_DIR')
  dl = build_dataloader(datasets_dir,
      ds, split=split, subset=subset, transform=transform,
      num_frames=num_frames, sampling=sampling, cache=cache,
      batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
      verbose=verbose, print_dropped=print_dropped)

  print(f'Number of batches {len(dl)}')
  for epoch in range(epochs):
    print(f'Epoch {epoch}')
    for x, y in islice(dl, batches):
      print(
        f"  x {x.dtype} {x.shape}\n"
        f"  x.flatten()[:5] {x.flatten()[:5]}\n"
        f"  y {y.dtype} {y.shape}\n"
        f"  y[0] {y[0]}"
      )
      if plot:
        for i in range(len(x)):
          clip = [x[i][j] for j in range(len(x[i]))]
          plot_clip(clip)

if __name__ == '__main__':
  fire.Fire(test)
