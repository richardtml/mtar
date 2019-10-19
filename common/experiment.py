""" exp.py

Base experiment class.
"""

import json
import os


class BaseExperiment:
  """Base experiment class."""

  def init_params(self, params):
    """Adds params as object attributes."""
    params = {att: val for att, val in params.items()
        if att not in ('self', 'args', 'kwargs')}
    # print(params)
    for att, val in params.items():
        setattr(self, att, val)

  def save_params(self, exp_dir):
    """Saves params in experiment.json"""
    params = {k: v for k, v in self.__dict__.items() if k[0] != '_'}
    params_path = os.path.join(exp_dir, 'experiment.json')
    os.makedirs(exp_dir, exist_ok=True)
    with open(params_path, 'w') as f:
      json.dump(params, f, indent=2, sort_keys=True)
