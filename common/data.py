""" data.py
"""

from itertools import islice

from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader


class DataLoader(TorchDataLoader):
  """Enriched wrapper dataloader."""

  def __iter__(self):
    """Returns an iterable with torch.Tensor's converted to np.ndarray's."""
    return ((z.numpy() if isinstance(z, Tensor) else z for z in batch)
      for batch in TorchDataLoader.__iter__(self))

  def first(self):
    """Returns the first batch."""
    return next(iter(self))

  def take(self, n):
    """Returns an iterable with the first `n` batches."""
    return islice(self, n)
