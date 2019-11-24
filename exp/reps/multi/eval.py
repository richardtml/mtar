"""eval.py
"""

import glob
import json
import os
import shutil
from collections import namedtuple
from functools import partial
from itertools import zip_longest
from os.path import exists, isdir, join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import fire
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm, trange

import models
import utils
from data import build_dataloader
from common import config
from exp.experiment import BaseExperiment


tzip = partial(zip_longest, fillvalue=(None, None))


def build_tasks_eval(datasets_dir, run_dir, batch_size, cfg):
  """Builds tasks evaluation object."""
  TasksEval = namedtuple('TasksEval', ('etrn', 'etst'))
  Subset = namedtuple('Subset', ('tasks', 'writer'))
  Task = namedtuple('Task', ('name', 'dl', 'loss', 'acc'))
  subsets = []
  for alias, name in zip(('etrn', 'etst'), ('train', 'test')):
    tasks = []
    for ds in cfg._dss:
      dl = build_dataloader(datasets_dir=datasets_dir,
          ds=ds.name, split=ds.split, subset=name,
          batch_size=batch_size, cache=True)
      loss = tf.keras.metrics.SparseCategoricalCrossentropy()
      acc = tf.keras.metrics.SparseCategoricalAccuracy()
      tasks.append(Task(ds.name, dl, loss, acc))
    writer = tf.summary.create_file_writer(join(run_dir, alias))
    subsets.append(Subset(tasks, writer))
  tasks_eval = TasksEval(*subsets)
  return tasks_eval


def load_model(run_dir, cfg, epoch):
  weights_dir = join(run_dir, 'weights')
  if epoch is not None:
    checkpoint = join(weights_dir, f'{epoch:03d}.ckpt')
  else:
    checkpoint = tf.train.latest_checkpoint(weights_dir)
  ModelClass = models.get_model_class(cfg.model)
  model = ModelClass(cfg)
  model([np.zeros((1, 16, 512), dtype=np.float32) for _ in cfg._dss])
  model.load_weights(checkpoint)
  return model

@tf.function
def call_model(model, xs):
  return model(xs)

def eval_subset(model, subset):
  metrics_fns = [
    [tf.keras.metrics.SparseCategoricalCrossentropy(),
    tf.keras.metrics.SparseCategoricalAccuracy()]
    for _ in subset.tasks
  ]
  for batches in tzip(*[task.dl for task in subset.tasks]):
    xs, ys_true = zip(*batches)
    # ys_pred = model(xs)
    ys_pred = call_model(model, xs)
    for y_true, y_pred, (loss_fn, acc_fn) in zip(ys_true, ys_pred, metrics_fns):
      if y_true is not None:
        loss_fn(y_true, y_pred)
        acc_fn(y_true, y_pred)
  metrics = [
    [loss_fn.result().numpy()*100, acc_fn.result().numpy()*100]
    for loss_fn, acc_fn in metrics_fns
  ]
  return metrics


def eval_run(run_dir, batch_size=128, epoch=None,
    verbose=True, tqdm_leave=True, tqdm_disable=False):

  cfg = BaseExperiment.load(run_dir)
  cfg._dss = utils.build_datasets(cfg)
  datasets_dir = config.get('DATASETS_DIR')
  tasks_eval = build_tasks_eval(datasets_dir, run_dir, batch_size, cfg)

  if epoch is not None:
    print(f'Evaluating {cfg.run} at epoch {epoch}')
    model = load_model(run_dir, cfg, epoch)
    metrics_etrn = eval_subset(model, tasks_eval.etrn)
    metrics_etst = eval_subset(model, tasks_eval.etst)
    for ds, m_etrn, m_etst in zip(cfg._dss, metrics_etrn, metrics_etst):
      loss_etrn, acc_etrn = m_etrn
      loss_etst, acc_etst = m_etst
      print(
        f'{ds.name}'
        f' loss=({loss_etrn:.2f},{loss_etst:.2f})'
        f' acc=({acc_etrn:.2f},{acc_etst:.2f})'
      )
    return

  if verbose:
    print(f'Evaluating {cfg.run}')
  bests = [[0, 0] for _ in cfg._dss]
  tasks_sets = (tasks_eval.etrn, tasks_eval.etst)
  for epoch in trange(cfg.train_epochs,
      leave=tqdm_leave, disable=tqdm_disable):
  # for epoch in trange(2, leave=tqdm_leave):
    model = load_model(run_dir, cfg, epoch)
    etrn = eval_subset(model, tasks_eval.etrn)
    etst = eval_subset(model, tasks_eval.etst)
    for sname, teval, mset in zip(('etrn', 'etst'), tasks_sets, (etrn, etst)):
      with teval.writer.as_default():
        for tidx, (task, (loss, acc)) in enumerate(zip(teval.tasks, mset)):
          # print(sname, task.name, 'loss', loss, epoch)
          # print(sname, task.name, 'acc ', acc, epoch)
          tf.summary.scalar(f'loss/{task.name}', loss, epoch)
          tf.summary.scalar(f'acc/{task.name}', acc, epoch)
        # test
          if sname == 'etst' and bests[tidx][0] < acc:
              bests[tidx] = [acc, epoch]

  data = dict(cfg)
  data['opt_alphas'] = str(data['opt_alphas'])
  columns = ['run']
  for ds, best in zip(cfg._dss, bests):
    acc, epoch = best
    col_acc, col_epoch = f'{ds.name}_acc', f'{ds.name}_epoch'
    columns.extend([col_acc, col_epoch])
    data[col_acc] = acc
    data[col_epoch] = epoch
  columns.extend([k for k in cfg.keys() if k not in ['run', '_dss']])
  df = pd.DataFrame(data, columns=columns, index=[0])
  df.to_csv(f'{run_dir}/results.csv', index=False)
  if verbose:
    print(df.head())


def filter_runs_dirs(runs_dirs):
  evaluable_runs_dirs = []
  for run_dir in runs_dirs:
    if isdir(run_dir):
      cfg = BaseExperiment.load(run_dir)
      trained_epochs = len(glob.glob(join(run_dir, 'weights', '*.index')))
      if trained_epochs == cfg.train_epochs:
        if not exists(join(run_dir, 'results.csv')):
          evaluable_runs_dirs.append(run_dir)
  return evaluable_runs_dirs


# def eval_exp(exp_dir, batch_size=128):
#   processed_results = pd.read_csv(f'{exp_dir}/results.csv', index_col=0)
#   dfs = [processed_results]
#   runs_dirs = [join(exp_dir, d) for d in os.listdir(exp_dir)]
#   runs_dirs = filter_runs_dirs(os.listdir(exp_dir))
#   for run_dir in tqdm(runs_dirs):
#     eval_run(run_dir, batch_size, epoch=None,
#         verbose=False, tqdm_leave=False)
#     run_df = pd.read_csv(join(run_dir, 'results.csv'), index_col=0)
#     dfs.append(run_df)
#     df = pd.concat(dfs, sort=False)
#     df.to_csv(f'{exp_dir}/results.csv')


def eval_exp(exp_dir, batch_size=128):
  import subprocess

  dfs = []
  if os.path.exists(f'{exp_dir}/results.csv'):
    dfs.append(results = pd.read_csv(f'{exp_dir}/results.csv', index_col=0))
  runs_dirs = sorted([join(exp_dir, d) for d in os.listdir(exp_dir)])
  runs_dirs = filter_runs_dirs(runs_dirs)
  print(runs_dirs)
  for run_dir in tqdm(runs_dirs):
    cmd = (
      f'python eval.py run {run_dir}'
      f' --batch_size {batch_size}'
      f' --epoch {None}'
      f' --verbose {False}'
      f' --tqdm_leave {True}'
    )
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
      print(f'Failed: python eval.py run {run_dir}')
    run_df = pd.read_csv(join(run_dir, 'results.csv'), index_col=0)
    dfs.append(run_df)
    df = pd.concat(dfs, sort=False)
    df.to_csv(f'{exp_dir}/results.csv')

if __name__ == '__main__':
  fire.Fire({
    'exp': eval_exp,
    'run': eval_run
  })
