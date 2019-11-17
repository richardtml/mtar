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
from tqdm import trange

import models
from common import config
from common.utils import timestamp
from data import build_dataloader
from exp.experiment import BaseExperiment


seed = config.get('SEED')
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
torch.manual_seed(seed)


def build_trn_dls(datasets_dir, cfg):
  """Builds training datases."""
  dls = []
  for task_name, task_prop in cfg.tasks.items():
    dl = build_dataloader(datasets_dir, task_name,
        task_prop['split'], 'train', cfg.train['tbatch'])
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
    for task_name, task_prop in cfg.tasks.items():
      dl = build_dataloader(datasets_dir, task_name,
          task_prop['split'], name, cfg.train['ebatch'])
      loss = tf.keras.metrics.SparseCategoricalCrossentropy()
      acc = tf.keras.metrics.SparseCategoricalAccuracy()
      tasks.append(Task(task_name, dl, loss, acc))
    writer = tf.summary.create_file_writer(join(run_dir, alias))
    subsets.append(Subset(tasks, writer))
  tasks_eval = TasksEval(*subsets)
  return tasks_eval

def build_loss_opt(cfg):
  """Builds loss and optimization objects."""
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
  opt = tf.keras.optimizers.SGD(learning_rate=cfg.train.lr,
    momentum=cfg.train.momentum, nesterov=cfg.train.nesterov)
  return loss_fn, opt

def tzip_refill(*dls):
  sizes = [len(dl) for dl in dls]
  i_max_dl = np.argmax(sizes)
  its = [iter(dl) for dl in dls]
  while True:
    batches = []
    for i in range(len(its)):
      try:
        batch = next(its[i])
        batches.append(batch)
      except StopIteration:
        if i != i_max_dl:
          its[i] = iter(dls[i])
          batch = next(its[i])
          batches.append(batch)
        else:
          return
    yield batches

def tzip_interleave(*dls, fillvalue=(None, None)):
  sizes = np.array([len(dl) for dl in dls])
  max_size = np.max(sizes)
  steps = [max_size // size for size in sizes]
  masks = [np.zeros(max_size) for _ in sizes]
  for mask, size, step in zip(masks, sizes, steps):
    indices = np.arange(0, size*step, step)
    remaining = (max_size - 1) - indices[-1]
    if remaining:
      #print('indices', indices)
      #print('remaining', remaining)
      offset = np.arange(remaining)
      #print('offset', offset)
      if max_size % size:
        indices[-len(offset):] = indices[-len(offset):] + offset[:len(indices)]
    np.put(mask, indices, np.ones_like(indices))
  its = [iter(dl) for dl in dls]
  for batches_mask in zip(*masks):
    batches = []
    for batch_mask, it in zip(batches_mask, its):
      if batch_mask:
        batch = next(it)
      else:
        batch = fillvalue
      batches.append(batch)
    yield batches

def build_tzip(cfg):
  """Builds training strategy zip."""
  strategy = cfg.train.strategy
  if strategy == 'shortest':
    return zip
  elif strategy == 'longest':
    return partial(zip_longest, fillvalue=(None, None))
  elif strategy == 'refill':
    return tzip_refill
  elif strategy == 'interleave':
    return tzip_interleave
  else:
    raise ValueError(f'invalid cfg.train.strategy={strategy}')

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

def eval_step_subset(model, subset):
  batches = [task.dl.first() for task in subset.tasks]
  xs, ys_true = zip(*batches)
  ys_pred = model(xs)
  for y_true, y_pred, task in zip(ys_true, ys_pred, subset.tasks):
    task.loss(y_true, y_pred)
    task.acc(y_true, y_pred)

@tf.function
def eval_step(model, tasks_eval):
  eval_step_subset(model, tasks_eval.trn)
  eval_step_subset(model, tasks_eval.tst)

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

  ModelClass = models.get_model_class(cfg.model.name)
  model = ModelClass(cfg)

  datasets_dir = config.get('DATASETS_DIR')
  trn_dls = build_trn_dls(datasets_dir, cfg)
  tzip = build_tzip(cfg)
  loss_fn, opt = build_loss_opt(cfg)
  tasks_eval = build_tasks_eval(datasets_dir, run_dir, cfg)

  epochs = cfg.train['epochs']
  alphas = cfg.train['alphas']
  weights_dir = join(run_dir, 'weights')
  for epoch in trange(epochs):
    for batches in tzip(*trn_dls):
      xs, ys_true = zip(*batches)
      train_step(xs, ys_true, model, loss_fn, opt, alphas)
      eval_step(model, tasks_eval)
    eval_epoch(epoch, tasks_eval)
    model.save_weights(join(weights_dir, f'{epoch:03d}.ckpt'))


class Run(BaseExperiment):

  def __init__(self,
      exp='msframe',
      model={
        'name': 'SFrame',
        'bn_in': 0,
        'conv1d': 128,
        'conv2d': 128,
        'dropout': 0.5,
        # 'rec_type': 'gru',
        # 'rec_size': 128,
      },
      tasks={
        'hmdb51': {'split': 1, 'bn_out': 0},
        'ucf101': {'split': 1, 'bn_out': 0},
      },
      train={
        'strategy': 'interleave',
        'epochs': 10,
        'lr': 1e-2,
        'optimizer': 'sgd',
        'momentum': 0.0,
        'nesterov': False,
        'tbatch': 128,
        'ebatch': 64,
        'alphas': [1, 1],
      }
    ):
    self.update(
      {k: v for k, v in reversed(list(locals().items())) if k != 'self'}
    )

  def __call__(self):
    self.run = f'{timestamp()}-{self.model.name}'
    train(self)


if __name__ == '__main__':
  fire.Fire(Run)
