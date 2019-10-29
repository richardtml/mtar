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
      batch_shape=(2, 1, 16, 512),
      reps_size = 512,
      batchnorm=True,
      conv2d_filters=128,
      dropout=0.5,
      rec_type = 'gru',
      rec_size = 128,
      ):
    self.init_params(locals())

  def __call__(self, model):
    test(model, self)


def test(class_name, cfg):
  """Test a model printing batch flow in call() method."""
  ModelClass = get_model_class(class_name)
  model = ModelClass(cfg, verbose=True)
  model.build(input_shape=tuple(cfg.batch_shape))
  model.summary()


if __name__ == '__main__':
  fire.Fire(Experiment)
