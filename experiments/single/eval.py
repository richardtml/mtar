"""eval.py
"""

from datetime import datetime
import json
import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import fire
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import models
from data.ucf101 import build_dataloader
from common import config


DATASETS_DIR = config.get('DATASETS_DIR')


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self


def parse_dir_name(dir_name):
  dir_name = dir_name.strip('/')
  model_id = dir_name.split('/')[-1]
  ts, ds_name, model_class_name = model_id.split('-')
  return model_id, ts, ds_name, model_class_name


def load_config(model_dir):
  json_path = join(model_dir, 'experiment.json')
  with open(json_path, 'r') as f:
    json_dict = json.load(f)
    cfg = Config()
    cfg.update(json_dict)
    return cfg


def load_model(model_dir, cfg, epoch):
  model_id, _, _, model_class_name = parse_dir_name(model_dir)
  weights_dir = join(model_dir, 'weights')
  if epoch:
    checkpoint = join(weights_dir, f'{epoch:03d}.ckpt')
  else:
    checkpoint = tf.train.latest_checkpoint(weights_dir)
  ckp = checkpoint.split('/')[-1]
  print(f'Loading {model_id} with {ckp}')
  model_class = models.get_model_class(model_class_name)
  model = model_class(cfg)
  model(np.zeros((1, 16, cfg.reps_size), dtype=np.float32))
  model.load_weights(checkpoint)
  return model


def eval_set(model, dl):
  loss_epoch = tf.keras.metrics.SparseCategoricalCrossentropy()
  acc_epoch = tf.keras.metrics.SparseCategoricalAccuracy()
  for x, y_true in tqdm(dl):
    y_pred = model(x)
    loss_epoch(y_true, y_pred)
    acc_epoch(y_true, y_pred)
  loss = loss_epoch.result().numpy() * 100
  acc = acc_epoch.result().numpy() * 100
  return loss, acc


def eval(model_dir, epoch=None, batch_size=128):
  cfg = load_config(model_dir)

  ds_dir = join(DATASETS_DIR, cfg.ds)
  trn_dl = build_dataloader(ds_dir, 'train', cfg.split, batch_size)
  tst_dl = build_dataloader(ds_dir, 'test', cfg.split, batch_size)

  model = load_model(model_dir, cfg, epoch)

  loss, acc = eval_set(model, trn_dl)
  print(f'trn_loss {loss}')
  print(f'trn_acc {acc}')

  loss, acc = eval_set(model, tst_dl)
  print(f'tst_loss {loss}')
  print(f'tst_acc {acc}')


if __name__ == '__main__':
  fire.Fire(eval)
