"""eval.py
"""

import glob
import json
import os
import shutil
from os.path import isdir, join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import fire
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm, trange

import models
from data import build_dataloader
from common import config


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_class(dir_name):
  dir_name = dir_name.strip('/')
  run = dir_name.split('/')[-1]
  _, _, model_class_name = run.split('-')
  return model_class_name

def load_config(run_dir):
  json_path = join(run_dir, 'experiment.json')
  with open(json_path, 'r') as f:
    json_dict = json.load(f)
    cfg = Config()
    cfg.update(json_dict)
    return cfg

def load_model(run_dir, cfg, epoch):
  model_class_name = get_class(run_dir)
  weights_dir = join(run_dir, 'weights')
  if epoch:
    checkpoint = join(weights_dir, f'{epoch:03d}.ckpt')
  else:
    checkpoint = tf.train.latest_checkpoint(weights_dir)
  num_classes = 51 if cfg.ds == 'hmdb51' else 101
  ModelClass = models.get_model_class(model_class_name)
  model = ModelClass(cfg, num_classes)
  model(np.zeros((1, 16, cfg.reps_size), dtype=np.float32))
  model.load_weights(checkpoint)
  return model

def eval_subset(model, dl):
  acc_fn = tf.keras.metrics.SparseCategoricalAccuracy()
  for x, y_true in dl:
    y_pred = model(x)
    acc_fn(y_true, y_pred)
  acc = acc_fn.result().numpy() * 100
  return acc


def eval_run(run_dir, batch_size=128, epoch=None,
    leave=True, trn_dl=None, tst_dl=None, verbose=True):
  cfg = load_config(run_dir)
  if not trn_dl:
    datasets_dir = config.get('DATASETS_DIR')
    trn_dl = build_dataloader(datasets_dir,
        cfg.ds, 'train', cfg.split, batch_size)
    tst_dl = build_dataloader(datasets_dir,
        cfg.ds, 'test', cfg.split, batch_size)

  if epoch:
    print(f'Evaluating {cfg.run} at epoch {epoch}')
    model = load_model(run_dir, cfg, epoch)
    trn_acc = eval_subset(model, trn_dl)
    tst_acc = eval_subset(model, tst_dl)
    print(f'{cfg.run} acc {trn_acc} {tst_acc}')
    return

  trn_dir = join(run_dir, 'etrn')
  tst_dir = join(run_dir, 'etst')
  if isdir(trn_dir):
    shutil.rmtree(trn_dir)
  if isdir(tst_dir):
    shutil.rmtree(tst_dir)
  trn_writer = tf.summary.create_file_writer(trn_dir)
  tst_writer = tf.summary.create_file_writer(tst_dir)
  best_acc, best_epoch = 0, 0
  if verbose:
    print(f'Evaluating {cfg.run}')
  for epoch in trange(cfg.epochs, leave=leave):
    model = load_model(run_dir, cfg, epoch)
    trn_acc = eval_subset(model, trn_dl)
    tst_acc = eval_subset(model, tst_dl)
    if trn_acc > best_acc:
      best_acc, best_epoch = trn_acc, epoch
    with trn_writer.as_default():
      tf.summary.scalar(f'acc/{cfg.ds}', trn_acc, epoch)
    with tst_writer.as_default():
      tf.summary.scalar(f'acc/{cfg.ds}', tst_acc, epoch)

  firsts = ['run', 'ds', 'split']
  columns = [k for k in sorted(cfg.keys()) if k not in firsts]
  columns = firsts + ['acc', 'epoch'] + columns
  data = dict(cfg)
  data['acc'] = best_acc
  data['epoch'] = best_epoch
  df = pd.DataFrame(data, columns=columns, index=[0])
  df.to_csv(f'{run_dir}/results.csv')
  if verbose:
    print(df.head())

def eval_exp(exp_dir, batch_size=128):
  runs_names = sorted(os.listdir(exp_dir))
  run_dir = join(exp_dir, runs_names[0])
  datasets_dir = config.get('DATASETS_DIR')
  cfg = load_config(run_dir)
  trn_dl = build_dataloader(datasets_dir,
      cfg.ds, 'train', cfg.split, batch_size)
  tst_dl = build_dataloader(datasets_dir,
      cfg.ds, 'test', cfg.split, batch_size)

  dfs = []
  for run_name in tqdm(runs_names):
    run_dir = join(exp_dir, run_name)
    if isdir(run_dir):
      eval_run(run_dir, batch_size, epoch=None, leave=False,
          trn_dl=trn_dl, tst_dl=tst_dl, verbose=False)
      run_df = pd.read_csv(join(run_dir, 'results.csv'))
      dfs.append(run_df)
      df = pd.concat(dfs)
      df.to_csv(f'{exp_dir}/results.csv')


if __name__ == '__main__':
  fire.Fire({
    'exp': eval_exp,
    'run': eval_run
  })
