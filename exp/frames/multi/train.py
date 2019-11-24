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

from common import config
from common.utils import timestamp
from data import build_dataloader
from experiment.experiment import BaseExperiment
import models
import utils


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
        ds=ds.name, split=ds.split, subset='train',
        batch_size=cfg.train_tbatch, sampling=cfg.dss_sampling,
        cache=cfg.dss_cache)
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
          batch_size=cfg.train_ebatch, cache=False)
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
  strategy = cfg.train_strategy
  if strategy == 'shortest':
    return zip
  elif strategy == 'longest':
    return partial(zip_longest, fillvalue=(None, None))
  elif strategy == 'refill':
    return tzip_refill
  elif strategy == 'interleave':
    return tzip_interleave
  else:
    raise ValueError(f'invalid param train_strategy={strategy}')

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


  extractor = models.load_cnn_extractor()
  ModelClass = models.get_model_class(cfg.model)
  model = ModelClass(cfg)

  datasets_dir = config.get('DATASETS_DIR')
  trn_dls = build_trn_dls(datasets_dir, cfg)
  tzip = build_tzip(cfg)
  loss_fn, opt = build_loss_opt(cfg)
  tasks_eval = build_tasks_eval(datasets_dir, run_dir, cfg)

  weights_dir = join(run_dir, 'weights')
  for epoch in trange(cfg.train_epochs):
    for batches in tqdm(tzip(*trn_dls)):
      xs, ys_true = zip(*batches)
      xs = compute_reps(xs, extractor)
      train_step(xs, ys_true, model, loss_fn, opt, cfg.opt_alphas)
    #   eval_step(model, tasks_eval)
    # eval_epoch(epoch, tasks_eval)
    # model.save_weights(join(weights_dir, f'{epoch:03d}.ckpt'))


class RunConfig(BaseExperiment):

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
      train_epochs=2,
      train_tbatch=32,
      train_ebatch=16,
      # optimizer
      opt='sgd',
      opt_lr=1e-2,
      opt_momentum=0.0,
      opt_nesterov=False,
      opt_alphas=[1, 1],
      # cache
      dss_sampling='fixed',
      dss_cache=False,
      dss_num_workers=0,
    ):
    run = f'{timestamp()}-{model}-{train_strategy}'
    self.update(
      {k: v for k, v in reversed(list(locals().items())) if k != 'self'}
    )

  def __call__(self):
    self._dss = utils.build_datasets(self)
    train(self)


if __name__ == '__main__':
  fire.Fire(RunConfig)
