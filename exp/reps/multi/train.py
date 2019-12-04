""" train.py
"""

import os
import random
from collections import namedtuple
from functools import partial
from itertools import zip_longest
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import fire
import mlflow
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm, trange

import models
import utils
from common import config
from common.utils import timestamp
from data import build_dataloader


seed = config.get('SEED')
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
torch.manual_seed(seed)

TQDM_NCOLS = 75

class SubsetContext:

  def __init__(self, name, full_name, datasets_dir, run_dir, cfg):
    self.writer = tf.summary.create_file_writer(join(run_dir, name))
    self.tasks = []
    batch_size = cfg.train_batch // len(cfg._dss)
    Task = namedtuple('Task', ('name', 'dl', 'loss', 'acc'))
    if name == 'trn':
      sampling = cfg.dss_sampling
      shuffle = True
    else:
      sampling = 'fixed'
      shuffle = False
    for ds in cfg._dss:
      dl = build_dataloader(datasets_dir=datasets_dir,
        ds=ds.name, split=ds.split, subset=full_name,
        sampling=sampling, cache=cfg.dss_cache, batch_size=batch_size,
        shuffle=shuffle, num_workers=cfg.dss_num_workers)
      loss = tf.keras.metrics.SparseCategoricalCrossentropy()
      acc = tf.keras.metrics.SparseCategoricalAccuracy()
      self.tasks.append(Task(ds.name, dl, loss, acc))

    self.dls = [task.dl for task in self.tasks]

def build_subsets_contexts(datasets_dir, run_dir, cfg):
  return [SubsetContext('trn', 'train', datasets_dir, run_dir, cfg),
          SubsetContext('tst', 'test', datasets_dir, run_dir, cfg)]

def build_loss_opt(cfg):
  """Builds loss and optimization objects."""
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
  if cfg.opt == 'sgd':
    opt = tf.keras.optimizers.SGD(learning_rate=cfg.opt_lr,
      momentum=cfg.opt_momentum, nesterov=cfg.opt_nesterov)
  elif cfg.opt == 'adam':
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.opt_lr)
  else:
    raise ValueError(f'invalid opt={cfg.opt}')
  return loss_fn, opt

@tf.function
def train_step(xs, ys_true, model, loss_fn, opt, alphas, ctx):
  with tf.GradientTape() as tape:
    ys_pred = model(xs, training=True)
    losses = []
    for y_true, y_pred, alpha, task in zip(
        ys_true, ys_pred, alphas, ctx.tasks):
      if y_true is not None:
        losses.append(loss_fn(y_true, y_pred) * alpha)
        task.loss(y_true, y_pred)
        task.acc(y_true, y_pred)
    loss = tf.reduce_sum(losses)
  gradients = tape.gradient(loss, model.trainable_variables)
  opt.apply_gradients(zip(gradients, model.trainable_variables))

def train_epoch(epoch, tzip, model, loss_fn, opt, alphas, ctx):
  for batches in tqdm(tzip(*ctx.dls), leave=False,
      desc='  train', ncols=TQDM_NCOLS):
    xs, ys_true = zip(*batches)
    train_step(xs, ys_true, model, loss_fn, opt, alphas, ctx)

@tf.function
def test_step(xs, ys_true, model, ctx):
  ys_pred = model(xs)
  for y_true, y_pred, task in zip(ys_true, ys_pred, ctx.tasks):
    if y_true is not None:
      task.loss(y_true, y_pred)
      task.acc(y_true, y_pred)

def test_epoch(epoch, tzip, model, trn_ctx, tst_ctx):
  for batches in tqdm(tzip(*tst_ctx.dls), leave=False,
      desc='   test', ncols=TQDM_NCOLS):
    xs, ys_true = zip(*batches)
    test_step(xs, ys_true, model, tst_ctx)
  for ctx in (trn_ctx, tst_ctx):
    with ctx.writer.as_default():
      for task in ctx.tasks:
        loss = task.loss.result().numpy() * 100
        acc = task.acc.result().numpy() * 100
        task.loss.reset_states()
        task.acc.reset_states()
        tf.summary.scalar(f'loss/{task.name}', loss, epoch)
        tf.summary.scalar(f'acc/{task.name}', acc, epoch)

def train(cfg):
  print(f"Trainig {cfg.run}")
  run_dir = join(config.get('RESULTS_DIR'), cfg.exp, cfg.run)
  cfg.save(run_dir)

  ModelClass = models.get_model_class(cfg.model)
  model = ModelClass(cfg)

  datasets_dir = config.get('DATASETS_DIR')
  trn_ctx, tst_ctx = build_subsets_contexts(datasets_dir, run_dir, cfg)
  trn_zip = utils.build_tzip(cfg.train_strategy)
  tst_zip = utils.build_tzip('longest')
  loss_fn, opt = build_loss_opt(cfg)
  weights_dir = join(run_dir, 'weights')

  for epoch in trange(cfg.train_epochs, desc=' epochs', ncols=TQDM_NCOLS):

    train_epoch(epoch, trn_zip, model,
      loss_fn, opt, cfg.opt_alphas, trn_ctx)

    if cfg.eval_tst_freq and ((epoch + 1) % cfg.eval_tst_freq == 0):
      test_epoch(epoch, tst_zip, model, trn_ctx, tst_ctx)

    if cfg.save_freq and ((epoch + 1) % cfg.save_freq == 0):
      model.save_weights(join(weights_dir, f'{epoch:03d}.ckpt'))


class RunConfig(utils.BaseExperiment):

  def __init__(self,
      # experiment
      run=None,
      exp='reps',
      # model
      model='SFrame',
      model_bn_in=0,
      model_bn_out=0,
      model_rec_type='gru',
      model_rec_sizes=[128],
      model_rec_size=128,
      model_rec_layers=1,
      model_rec_bi=0,
      model_rec_bi_merge='concat',
      model_conv_filters=[128],
      model_conv2d_filters=160,
      model_conv1d_filters=160,
      model_dropout=0.5,
      model_ifc=0,
      # datasets
      hmdb51=False,
      hmdb51_split=1,
      ucf101=False,
      ucf101_split=1,
      # training
      train_strategy='shortest',
      train_epochs=1,
      train_batch=128,
      # optimizer
      opt='sgd',
      opt_lr=1e-3,
      opt_momentum=0.0,
      opt_nesterov=False,
      opt_alphas=[1, 1],
      # eval
      eval_trn_freq=1,
      eval_tst_freq=1,
      save_freq=0,
      # dss
      dss_sampling='fixed',
      dss_cache=False,
      dss_num_workers=1,
    ):
    self.update(
      {k: v for k, v in reversed(list(locals().items())) if k != 'self'}
    )

  def __call__(self):
    self._dss = utils.build_datasets_cfg(self)
    dss = ''.join([ds.name[0] for ds in self._dss])
    self.run = (
      f'{timestamp()}'
      f'-{dss}'
      f'-{self.model}'
      f'-{self.train_strategy[0]}'
      f'-{self.dss_sampling[0]}'
    )
    if len(self._dss) == 0:
      print('You must choose at least one task to train :).')
      return
    train(self)


if __name__ == '__main__':
  fire.Fire(RunConfig)
