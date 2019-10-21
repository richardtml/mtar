""" train.py
"""

import itertools as it
import os
from functools import partial
from itertools import zip_longest
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import fire
import numpy as np
import tensorflow as tf
from tqdm import trange

import models
from data import build_dataloader
from common import config
from common.experiment import BaseExperiment
from common.utils import timestamp


tf.random.set_seed(config.get('SEED'))


# def get_training_zip(strategy):
#   return (for zip_longest(g))

@tf.function
def train_step(x_h, y_true_h, x_u, y_true_u,
    model, loss_fn, optimizer, alphas):
  with tf.GradientTape() as tape:
    y_pred_h, y_pred_u = model((x_h, x_u), training=True)
    loss = 0
    if x_h is not None:
      loss += loss_fn(y_true_h, y_pred_h) * alphas[0]
    if x_u is not None:
      loss += loss_fn(y_true_u, y_pred_u) * alphas[1]
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# def train_step(x_h, y_true_h, x_u, y_true_u, model, loss_fn, optimizer):
#   print('--------------')
#   with tf.GradientTape() as tape:
#     y_pred_h, y_pred_u = model((x_h, x_u), training=True)
#     if (x_h is not None) and (x_u is not None):
#       loss = loss_fn(y_true_h, y_pred_h) + loss_fn(y_true_u, y_pred_u)
#     else:
#       if x_h is not None:
#         print(f'x_h.shape {x_h.shape}')
#         print(y_pred_u)
#         loss = loss_fn(y_true_h, y_pred_h)
#       if x_u is not None:
#         print(f'x_u.shape {x_u.shape}')
#         print(y_pred_h)
#         loss = loss_fn(y_true_u, y_pred_u)
#   gradients = tape.gradient(loss, model.trainable_variables)
#   optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def eval_step_set(model,
    dl_h, dl_u,
    loss_epoch_h, loss_epoch_u,
    acc_epoch_h, acc_epoch_u):
  x_h, y_true_h = dl_h.first()
  x_u, y_true_u = dl_u.first()
  y_pred_h, y_pred_u = model((x_h, x_u))
  loss_epoch_h(y_true_h, y_pred_h)
  loss_epoch_u(y_true_u, y_pred_u)
  acc_epoch_h(y_true_h, y_pred_h)
  acc_epoch_u(y_true_u, y_pred_u)

@tf.function
def eval_step(model, trn, tst):
  eval_step_set(model, *trn)
  eval_step_set(model, *tst)

def eval_epoch_set(epoch,
    loss_epoch_h, loss_epoch_u,
    acc_epoch_h, acc_epoch_u,
    writer):
  loss_h = loss_epoch_h.result().numpy() * 100
  loss_u = loss_epoch_u.result().numpy() * 100
  acc_h = acc_epoch_h.result().numpy() * 100
  acc_u = acc_epoch_u.result().numpy() * 100
  loss_epoch_h.reset_states()
  loss_epoch_u.reset_states()
  acc_epoch_h.reset_states()
  acc_epoch_u.reset_states()
  with writer.as_default():
    tf.summary.scalar('loss/hmdb51', loss_h, epoch)
    tf.summary.scalar('loss/ucf101', loss_u, epoch)
    tf.summary.scalar('acc/hmdb51', acc_h, epoch)
    tf.summary.scalar('acc/ucf101', acc_u, epoch)

# def eval_epoch_set(epoch, writer, subset,
#     loss_epoch_h, loss_epoch_u,
#     acc_epoch_h, acc_epoch_u,
#     ):
#   loss_h = loss_epoch_h.result().numpy() * 100
#   loss_u = loss_epoch_u.result().numpy() * 100
#   acc_h = acc_epoch_h.result().numpy() * 100
#   acc_u = acc_epoch_u.result().numpy() * 100
#   loss_epoch_h.reset_states()
#   loss_epoch_u.reset_states()
#   acc_epoch_h.reset_states()
#   acc_epoch_u.reset_states()
#   with writer.as_default():
#     tf.summary.scalar(f'loss/hmdb51/{subset}', loss_h, epoch)
#     tf.summary.scalar(f'loss/ucf101/{subset}', loss_u, epoch)
#     tf.summary.scalar(f'acc/hmdb51/{subset}', acc_h, epoch)
#     tf.summary.scalar(f'acc/ucf101/{subset}', acc_u, epoch)

def eval_epoch(epoch, trn, tst):
  eval_epoch_set(epoch, *trn)
  eval_epoch_set(epoch, *tst)

def train(cfg):
  print(f"Trainig {cfg.model_id}")
  cfg.save_params(cfg.experiment_dir)

  datasets_dir = config.get('DATASETS_DIR')
  trn_dl_h = build_dataloader(datasets_dir, 'hmdb51',
      'train', cfg.split, cfg.tbatch_size)
  etrn_dl_h = build_dataloader(datasets_dir, 'hmdb51',
      'train', cfg.split, cfg.ebatch_size)
  etst_dl_h = build_dataloader(datasets_dir, 'hmdb51',
      'test', cfg.split, cfg.ebatch_size)
  trn_dl_u = build_dataloader(datasets_dir, 'ucf101',
      'train', cfg.split, cfg.tbatch_size)
  etrn_dl_u = build_dataloader(datasets_dir, 'ucf101',
      'train', cfg.split, cfg.ebatch_size)
  etst_dl_u = build_dataloader(datasets_dir, 'ucf101',
      'test', cfg.split, cfg.ebatch_size)

  ModelClass = models.get_model_class(cfg.model_class_name)
  model = ModelClass(cfg)

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
  optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.lr)
  trn_loss_epoch_h = tf.keras.metrics.SparseCategoricalCrossentropy()
  trn_acc_epoch_h = tf.keras.metrics.SparseCategoricalAccuracy()
  tst_loss_epoch_h = tf.keras.metrics.SparseCategoricalCrossentropy()
  tst_acc_epoch_h = tf.keras.metrics.SparseCategoricalAccuracy()
  trn_loss_epoch_u = tf.keras.metrics.SparseCategoricalCrossentropy()
  trn_acc_epoch_u = tf.keras.metrics.SparseCategoricalAccuracy()
  tst_loss_epoch_u = tf.keras.metrics.SparseCategoricalCrossentropy()
  tst_acc_epoch_u = tf.keras.metrics.SparseCategoricalAccuracy()

  trn_writer = tf.summary.create_file_writer(join(cfg.experiment_dir, 'trn'))
  tst_writer = tf.summary.create_file_writer(join(cfg.experiment_dir, 'tst'))
  # writer = tf.summary.create_file_writer(cfg.experiment_dir)

  trn_eval_step = (
    etrn_dl_h, etrn_dl_u,
    trn_loss_epoch_h, trn_loss_epoch_u,
    trn_acc_epoch_h, trn_acc_epoch_u
  )
  tst_eval_step = (
    etst_dl_h, etst_dl_u,
    tst_loss_epoch_h, tst_loss_epoch_u,
    tst_acc_epoch_h, tst_acc_epoch_u
  )
  trn_eval_epoch = (
    trn_loss_epoch_h, trn_loss_epoch_u,
    trn_acc_epoch_h, trn_acc_epoch_u,
    trn_writer
  )
  tst_eval_epoch = (
    tst_loss_epoch_h, tst_loss_epoch_u,
    tst_acc_epoch_h, tst_acc_epoch_u,
    tst_writer
  )
  # trn_eval_epoch = (
  #   writer, 'trn',
  #   trn_loss_epoch_h, trn_loss_epoch_u,
  #   trn_acc_epoch_h, trn_acc_epoch_u
  # )
  # tst_eval_epoch = (
  #   writer, 'tst',
  #   tst_loss_epoch_h, tst_loss_epoch_u,
  #   tst_acc_epoch_h, tst_acc_epoch_u
  # )

  lzip = partial(zip_longest, fillvalue=(None, None))
  tzip = zip if cfg.train_strategy == 'shortest' else lzip

  weights_dir = join(cfg.experiment_dir, 'weights')
  for epoch in trange(cfg.epochs):
    # for (x_h, y_true_h), (x_u, y_true_u) in zip_longest(trn_dl_h, trn_dl_u):
    for (x_h, y_true_h), (x_u, y_true_u) in tzip(trn_dl_h, trn_dl_u):
      train_step(x_h, y_true_h, x_u, y_true_u,
          model, loss_fn, optimizer, cfg.alphas)
      eval_step(model, trn_eval_step, tst_eval_step)
    eval_epoch(epoch, trn_eval_epoch, tst_eval_epoch)
    model.save_weights(join(weights_dir, f'{epoch:03d}.ckpt'))


class Experiment(BaseExperiment):

  def __init__(self,
      split=1,
      reps_size=512,
      tbatch_size=64,
      ebatch_size=64,
      lr=1e-3,
      alphas=(1, 1),
      epochs=5,
      train_strategy='shortest', # shortest, longest
      conv2d_filters=128,
      dropout=0.5,
      batchnorm=True,
      rec_type='gru', # gru or lstm
      rec_size=128,
      exp_name='multi',
      ):
    self.init_params(locals())

  def __call__(self,
      model # Conv2D, Conv2D1D, FullConv Rec
      ):
    self.model_class_name = model
    self.model_id = f'{timestamp()}-{self.split}-{model}'
    self.experiment_dir = join(config.get('RESULTS_DIR'),
        self.exp_name, self.model_id)
    train(self)


if __name__ == '__main__':
  fire.Fire(Experiment)
