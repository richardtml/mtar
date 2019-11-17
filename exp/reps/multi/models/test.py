""" test.py

Simple test for models.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import fire
import numpy as np

from exp.reps.multi.exp.experiment import BaseExperiment
from utils import get_model_class


class Experiment(BaseExperiment):
  def __init__(self,
      model={
        'name': 'SFrame',
        'bn_in': 1,
        'conv1d': 128,
        'conv2d': 128,
        'dropout': 0.5,
        # 'rec_type': 'gru',
        # 'rec_size': 128,
      },
      tasks={
        'hmdb51': {'split': 1, 'bn_out': 1},
        'ucf101': {'split': 1, 'bn_out': 1},
      },
      batch=[1, 16, 512],
    ):
    params = {k: v for k, v in locals().items() if k != 'self'}
    self.update(params)

  def __call__(self):
    test(self)


def test(cfg):
  """Test a model printing batch flow in call() method."""
  print(cfg)
  Model = get_model_class(cfg.model.name)
  print(Model.__name__)
  # batch_shape = (len(cfg.tasks),) + tuple(cfg.batch)
  # batch = np.zeros(batch_shape, dtype=np.float32)
  batch = [np.zeros(cfg.batch, dtype=np.float32) for _ in range(len(cfg.tasks))]
  model = Model(cfg, verbose=True)
  model(batch)
  model.summary()


if __name__ == '__main__':
  fire.Fire(Experiment)
