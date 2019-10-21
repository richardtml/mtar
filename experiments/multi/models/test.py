""" test.py

Simple test for models.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import fire
import numpy as np

from common.experiment import BaseExperiment
from utils import get_model_class


class Experiment(BaseExperiment):
  def __init__(self,
      tasks=(True, True),
      batch_shape=(1, 16, 512),
      reps_size = 512,
      conv2d_filters=32,
      rec_type = 'gru',
      rec_size = 32,
      ):
    self.init_params(locals())

  def __call__(self, model):
    test(model, self)


def test(class_name, cfg):
  """Test a model printing batch flow in call() method."""
  x = np.zeros(cfg.batch_shape, dtype=np.float32)
  xs = [x if t else None for t in cfg.tasks]
  ModelClass = get_model_class(class_name)
  model = ModelClass(cfg, verbose=True)
  model(xs)


if __name__ == '__main__':
  fire.Fire(Experiment)
