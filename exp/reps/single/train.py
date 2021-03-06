""" train.py
"""

import itertools as it
import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import fire
import mlflow
import numpy as np
import tensorflow as tf
from tqdm import trange

import models
from data import build_dataloader
from common import config
from common.experiment import BaseExperiment
from common.utils import timestamp



@tf.function
def train_step(x, y_true, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
      y_pred = model(x, training=True)
      loss = loss_fn(y_true, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def eval_step_set(model, dl, loss_epoch, acc_epoch):
  x, y_true = dl.first()
  y_pred = model(x)
  loss_epoch(y_true, y_pred)
  acc_epoch(y_true, y_pred)

@tf.function
def eval_step(model, trn, tst):
  eval_step_set(model, *trn)
  eval_step_set(model, *tst)

def eval_epoch_set(epoch, ds, subset, loss_epoch, acc_epoch, writer):
  loss = loss_epoch.result().numpy() * 100
  acc = acc_epoch.result().numpy() * 100
  loss_epoch.reset_states()
  acc_epoch.reset_states()
  with writer.as_default():
    tf.summary.scalar(f'loss/{ds}', loss, epoch)
    tf.summary.scalar(f'acc/{ds}', acc, epoch)
  mlflow.log_metric(f'{subset}-loss-{ds}', loss, epoch)
  mlflow.log_metric(f'{subset}-acc-{ds}', acc, epoch)

def eval_epoch(epoch, ds, trn, tst):
  eval_epoch_set(epoch, ds, 'trn', *trn)
  eval_epoch_set(epoch, ds, 'tst', *tst)


def train_with_log(cfg):
  mlflow.set_tracking_uri(join(config.get('RESULTS_DIR'), 'mlruns'))
  mlflow.set_experiment(cfg.exp_name)
  with mlflow.start_run(run_name=cfg.run):
    params = {k: v for k, v in cfg.__dict__.items() if k[0] != '_'}
    for k, v in params.items():
      mlflow.log_param(k, v)
    train(cfg)


def train(cfg):
  model_dir = join(config.get('RESULTS_DIR'), cfg.exp_name, cfg.run)
  print(f"Trainig {cfg.run}")
  cfg.save_params(model_dir)

  datasets_dir = config.get('DATASETS_DIR')
  trn_dl = build_dataloader(datasets_dir, cfg.ds,
      cfg.split, 'train', cfg.tbatch_size)
  etrn_dl = build_dataloader(datasets_dir, cfg.ds,
      cfg.split, 'train', cfg.ebatch_size)
  etst_dl = build_dataloader(datasets_dir, cfg.ds,
      cfg.split, 'test', cfg.ebatch_size)

  num_classes = 51 if cfg.ds == 'hmdb51' else 101
  ModelClass = models.get_model_class(cfg.model)
  model = ModelClass(cfg, num_classes)

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
  optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.lr)
  trn_loss_epoch = tf.keras.metrics.SparseCategoricalCrossentropy()
  trn_acc_epoch = tf.keras.metrics.SparseCategoricalAccuracy()
  tst_loss_epoch = tf.keras.metrics.SparseCategoricalCrossentropy()
  tst_acc_epoch = tf.keras.metrics.SparseCategoricalAccuracy()

  trn_writer = tf.summary.create_file_writer(join(model_dir, 'trn'))
  tst_writer = tf.summary.create_file_writer(join(model_dir, 'tst'))

  trn_eval_step = (etrn_dl, trn_loss_epoch, trn_acc_epoch)
  tst_eval_step = (etst_dl, tst_loss_epoch, tst_acc_epoch)
  trn_eval_epoch = (trn_loss_epoch, trn_acc_epoch, trn_writer)
  tst_eval_epoch = (tst_loss_epoch, tst_acc_epoch, tst_writer)

  weights_dir = join(model_dir, 'weights')
  for epoch in trange(cfg.epochs):
    for x, y_true in trn_dl:
      train_step(x ,y_true, model, loss_fn, optimizer)
      eval_step(model, trn_eval_step, tst_eval_step)
    eval_epoch(epoch, cfg.ds, trn_eval_epoch, tst_eval_epoch)
    model.save_weights(join(weights_dir, f'{epoch:03d}.ckpt'))


class Experiment(BaseExperiment):

  def __init__(self,
      split=1,
      reps_size=512,
      tbatch_size=128,
      ebatch_size=64,
      lr=1e-3,
      epochs=5,
      conv2d_filters=128,
      conv1d_filters=128,
      dropout=0.5,
      rec_layers = 1,
      rec_type='gru', # gru or lstm
      rec_size=128,
      ifc=0,
      exp_name='single',
      ):
    self.init_params(locals())

  def __call__(self,
      ds, # hmdb51, ucf101
      model # Conv2D, Conv2D1D, FullConv, Rec
      ):
    self.ds = ds
    self.model = model
    self.run = f'{timestamp()}-{ds}-{model}'
    train_with_log(self)


if __name__ == '__main__':
  tf.random.set_seed(config.get('SEED'))
  fire.Fire(Experiment)
