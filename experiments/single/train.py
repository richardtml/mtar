""" train.py
"""

import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import fire
import numpy as np
import tensorflow as tf
from tqdm import trange

import models
from data.ucf101 import build_dataloader
from common import config
from common.experiment import BaseExperiment
from common.utils import timestamp


@tf.function
def train_step(x, y_true, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
      y_pred = model(x)
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

def eval_epoch_set(epoch, loss_epoch, acc_epoch, writer):
  loss = loss_epoch.result().numpy() * 100
  acc = acc_epoch.result().numpy() * 100
  loss_epoch.reset_states()
  acc_epoch.reset_states()
  with writer.as_default():
    tf.summary.scalar('loss', loss, epoch)
    tf.summary.scalar('acc', acc, epoch)

def eval_epoch(epoch, trn, tst):
  eval_epoch_set(epoch, *trn)
  eval_epoch_set(epoch, *tst)

def train(cfg):
  print(f"Trainig {cfg.model_id}")
  cfg.save_params(cfg.experiment_dir)

  ds_dir = join(config.get('DATASETS_DIR'), cfg.ds)
  trn_dl = build_dataloader(ds_dir,
      'train', cfg.split, cfg.tbatch_size)
  etrn_dl = build_dataloader(ds_dir,
      'train', cfg.split, cfg.ebatch_size)
  etst_dl = build_dataloader(ds_dir,
      'test', cfg.split, cfg.ebatch_size)

  model_class = models.get_model_class(cfg.model_class_name)
  model = model_class(cfg)

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
  optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.lr)
  trn_loss_epoch = tf.keras.metrics.SparseCategoricalCrossentropy()
  trn_acc_epoch = tf.keras.metrics.SparseCategoricalAccuracy()
  tst_loss_epoch = tf.keras.metrics.SparseCategoricalCrossentropy()
  tst_acc_epoch = tf.keras.metrics.SparseCategoricalAccuracy()

  trn_writer = tf.summary.create_file_writer(join(cfg.experiment_dir, 'trn'))
  tst_writer = tf.summary.create_file_writer(join(cfg.experiment_dir, 'tst'))

  trn_eval_step = (etrn_dl, trn_loss_epoch, trn_acc_epoch)
  tst_eval_step = (etst_dl, tst_loss_epoch, tst_acc_epoch)
  trn_eval_epoch = (trn_loss_epoch, trn_acc_epoch, trn_writer)
  tst_eval_epoch = (tst_loss_epoch, tst_acc_epoch, tst_writer)

  weights_dir = join(cfg.experiment_dir, 'weights')
  for epoch in trange(cfg.epochs):
    for x, y_true in trn_dl:
      train_step(x ,y_true, model, loss_fn, optimizer)
      eval_step(model, trn_eval_step, tst_eval_step)
    eval_epoch(epoch, trn_eval_epoch, tst_eval_epoch)
    model.save_weights(join(weights_dir, f'{epoch:03d}.ckpt'))


class Experiment(BaseExperiment):

  def __init__(self,
      split=1,
      reps_size=512,
      tbatch_size=64,
      ebatch_size=32,
      lr=1e-3,
      epochs=500,
      conv2d_filters=32,
      rec_type='gru', # gru or lstm
      rec_size=128,
      exp_name='single',
      ):
    self.init_params(locals())

  def __call__(self,
      ds, # ucf101
      model # ARConv, ARConv2D, ARConv2D1D, ARRec
      ):
    self.ds = ds
    self.model_class_name = model
    self.model_id = f'{timestamp()}-{ds}-{model}'
    self.experiment_dir = join(config.get('RESULTS_DIR'),
        self.exp_name, self.model_id)
    train(self)


if __name__ == '__main__':
  fire.Fire(Experiment)
