""" train.py
"""

import os
import random
from collections import namedtuple
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


def build_trn_dls(datasets_dir, cfg):
  """Builds training datases."""
  dls = []
  for ds in cfg._dss:
    dl = build_dataloader(datasets_dir=datasets_dir,
        ds=ds.name, split=ds.split, subset='train', transform=cfg.dss_augment,
        batch_size=cfg.train_tbatch, sampling=cfg.dss_sampling,
        cache=cfg.dss_cache, num_workers=cfg.dss_num_workers)
    dls.append(dl)
  return dls

def build_tasks_eval(datasets_dir, run_dir, cfg):
  """Builds tasks evaluation object."""
  TasksEval = namedtuple('TasksEval', ('trn', 'tst'))
  Subset = namedtuple('Subset', ('tasks', 'writer'))
  Task = namedtuple('Task', ('name', 'dl', 'loss', 'acc'))
  subsets = []
  for alias, name in zip(('trn', 'tst'), ('train', 'test')):
    tasks = []
    for ds in cfg._dss:
      dl = build_dataloader(datasets_dir=datasets_dir,
          ds=ds.name, split=ds.split, subset=name,
          batch_size=cfg.train_ebatch,
          cache=False, num_workers=cfg.dss_num_workers)
      loss = tf.keras.metrics.SparseCategoricalCrossentropy()
      acc = tf.keras.metrics.SparseCategoricalAccuracy()
      tasks.append(Task(ds.name, dl, loss, acc))
    writer = tf.summary.create_file_writer(join(run_dir, alias))
    subsets.append(Subset(tasks, writer))
  tasks_eval = TasksEval(*subsets)
  return tasks_eval

def build_loss_opt(cfg):
  """Builds loss and optimization objects."""
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
  opt = tf.keras.optimizers.SGD(learning_rate=cfg.opt_lr,
    momentum=cfg.opt_momentum, nesterov=cfg.opt_nesterov)
  return loss_fn, opt

@tf.function
def compute_reps(xs, extractor):
  reps = []
  for x in xs:
    if x is not None:
      x = x / 255
      n, f, h, w, c = tf.unstack(x.shape)
      x = tf.reshape(x, (n*f, h, w, c))
      x = extractor(x)
      x = tf.reshape(x, (n, f, -1))
      reps.append(x)
    else:
      reps.append(None)
  return reps

@tf.function
def train_step(xs, ys_true, model, loss_fn, opt, alphas):
  with tf.GradientTape() as tape:
    ys_pred = model(xs, training=True)
    losses = []
    for y_true, y_pred, alpha in zip(ys_true, ys_pred, alphas):
      if y_true is not None:
        losses.append(loss_fn(y_true, y_pred) * alpha)
    loss = tf.reduce_sum(losses)
  gradients = tape.gradient(loss, model.trainable_variables)
  opt.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def eval_step_subset_model(model, xs, ys_true, tasks):
  ys_pred = model(xs)
  for y_true, y_pred, task in zip(ys_true, ys_pred, tasks):
    task.loss(y_true, y_pred)
    task.acc(y_true, y_pred)

def eval_step_subset(extractor, model, subset):
  batches = [next(iter(task.dl)) for task in subset.tasks]
  batches = utils.batches_to_numpy(batches)
  xs, ys_true = zip(*batches)
  xs = compute_reps(xs, extractor)
  eval_step_subset_model(model, xs, ys_true, subset.tasks)

def eval_step(extractor, model, tasks_eval):
  eval_step_subset(extractor, model, tasks_eval.trn)
  eval_step_subset(extractor, model, tasks_eval.tst)

def eval_epoch_subset(epoch, subset):
  with subset.writer.as_default():
    for task in subset.tasks:
      loss = task.loss.result().numpy() * 100
      acc = task.acc.result().numpy() * 100
      task.loss.reset_states()
      task.acc.reset_states()
      tf.summary.scalar(f'loss/{task.name}', loss, epoch)
      tf.summary.scalar(f'acc/{task.name}', acc, epoch)
      # mlflow.log_metric(f'{subset_name}-loss-{task.name}', loss, epoch)
      # mlflow.log_metric(f'{subset_name}-acc-{task.name}', acc, epoch)

def eval_epoch(epoch, tasks_eval):
  eval_epoch_subset(epoch, tasks_eval.trn)
  eval_epoch_subset(epoch, tasks_eval.tst)

def train(cfg):
  print(f"Trainig {cfg.run}")
  run_dir = join(config.get('RESULTS_DIR'), cfg.exp, cfg.run)
  cfg.save(run_dir)


  extractor = models.load_cnn_extractor()
  ModelClass = models.get_model_class(cfg.model)
  model = ModelClass(cfg)

  datasets_dir = config.get('DATASETS_DIR')
  trn_dls = build_trn_dls(datasets_dir, cfg)
  tzip = utils.build_tzip(cfg)
  loss_fn, opt = build_loss_opt(cfg)
  tasks_eval = build_tasks_eval(datasets_dir, run_dir, cfg)

  weights_dir = join(run_dir, 'weights')
  for epoch in trange(cfg.train_epochs):
    for batches in tqdm(tzip(*trn_dls)):
      batches = utils.batches_to_numpy(batches)
      xs, ys_true = zip(*batches)
      xs = compute_reps(xs, extractor)
      train_step(xs, ys_true, model, loss_fn, opt, cfg.opt_alphas)
      eval_step(extractor, model, tasks_eval)
    eval_epoch(epoch, tasks_eval)
    model.save_weights(join(weights_dir, f'{epoch:03d}.ckpt'))


class RunConfig(utils.BaseExperiment):

  def __init__(self,
      # experiment
      run=None,
      exp='multi',
      # model
      model='MeanFC',
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
      # training
      train_strategy='shortest',
      train_epochs=1,
      train_tbatch=16,
      train_ebatch=8,
      # optimizer
      opt='sgd',
      opt_lr=1e-2,
      opt_momentum=0.0,
      opt_nesterov=False,
      opt_alphas=[1, 1],
      # cache
      dss_augment=0,
      dss_sampling='fixed',
      dss_cache=False,
      dss_num_workers=2,
    ):
    run = f'{timestamp()}-{model}-{train_strategy}'
    self.update(
      {k: v for k, v in reversed(list(locals().items())) if k != 'self'}
    )

  def __call__(self):
    self._dss = utils.build_datasets_cfg(self)
    train(self)


if __name__ == '__main__':
  fire.Fire(RunConfig)
