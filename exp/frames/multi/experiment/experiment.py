""" experiment.py

Experiment utils.
"""

import os
from collections import Mapping, MutableMapping
from collections.abc import ItemsView

import pandas as pd
import yaml


def filter_params(d):
  params = {}
  for k, v in d.items():
    if k[0] != '_':
      if isinstance(v, Mapping):
        params[k] = filter_params(v)
      else:
        params[k] = v
  return params


def flatten_dict(d):
  flat = {}
  for k, v in d.items():
    if isinstance(v, Mapping):
      for ki, vi in flatten_dict(v).items():
        flat[f'{k}_{ki}'] = vi
    else:
      flat[k] = v
  return flat


class ODict(MutableMapping):
  """Object that acts like a dict."""

  def __init__(self, *args, **kwargs):
    self.__dict__.update(*args, **kwargs)

  def __delitem__(self, k):
    del self.__dict__[k]

  def __getitem__(self, k):
    return self.__dict__[k]

  def __iter__(self):
    return iter(self.__dict__)

  def __len__(self):
    return len(self.__dict__)

  def __repr__(self):
    return str(self.__dict__)

  def __setitem__(self, k, v):
    self.__dict__[k] = ODict(v) if isinstance(v, Mapping) else v

  def __str__(self):
    return str(self.__dict__)


class BaseExperiment(ODict):
  """Base experiment class."""

  @staticmethod
  def load(run_dir, filename='run'):
    filepath = os.path.join(run_dir, f'{filename}.yaml')
    with open(filepath, 'r') as f:
      config = yaml.load(f, Loader=yaml.FullLoader)
      exp = BaseExperiment()
      exp.update(config)
      return exp

  def params(self):
    return filter_params(self)

  def flat_params(self):
    return flatten_dict(self.params())

  def save(self, run_dir, filename='run'):
    params = self.params()
    os.makedirs(run_dir, exist_ok=True)
    filepath = os.path.join(run_dir, f'{filename}.yaml')
    with open(filepath, 'w') as f:
      yaml.dump(params, f, sort_keys=False)


