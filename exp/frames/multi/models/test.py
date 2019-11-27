""" test.py

Simple test for models.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import fire
import numpy as np

from exp.reps.multi import utils
from exp.reps.multi.exp.experiment import BaseExperiment
from utils import get_model_class


class RunConfig(BaseExperiment):

  def __init__(self,
      batch=(1, 16, 512),
      # experiment
      run=None,
      exp='msframe',
      # model
      model='SFrame',
      model_bn_in=0,
      model_bn_out=0,
      model_rec_type='gru',
      model_rec_size=128,
      model_rec_layers=1,
      model_rec_bi=0,
      model_rec_bi_merge='concat',
      model_conv2d_filters=160,
      model_conv1d_filters=160,
      model_dropout=0.5,
      model_ifc=0,
      # datasets
      hmdb51=True,
      hmdb51_split=1,
      ucf101=True,
      ucf101_split=1,
    ):
    self.update(
      {k: v for k, v in reversed(list(locals().items())) if k != 'self'}
    )

  def __call__(self):
    self._dss = utils.build_datasets(self)
    test(self)


def test(cfg):
  """Test a model printing batch flow in call() method."""
  print(cfg)
  Model = get_model_class(cfg.model)
  print(Model.__name__)
  xs = [np.zeros(cfg.batch, dtype=np.float32) for _ in range(len(cfg._dss))]
  model = Model(cfg, verbose=True)
  ys = model(xs)
  for i, y in enumerate(ys):
    print(
      f"  ys[{i}] {y.dtype} {y.shape}\n"
      f"  ys[{i}] {y.numpy().flatten()[:5]}"
    )
  model.summary()


if __name__ == '__main__':
  fire.Fire(RunConfig)
