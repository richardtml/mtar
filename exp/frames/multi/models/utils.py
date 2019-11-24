""" utils.py

Models utils.
"""

import sys
from exp.frames.multi.models import models

import fire
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def get_model_class(class_name):
  try:
    return getattr(models, class_name)
  except AttributeError as ex:
    raise AttributeError(f'model class "{class_name}" does not exist') from ex


def load_cnn_extractor(arch='resnet50', shape=(224, 224), trainable=False):
  """Loads CNN model pretrained with Imagenet without top layers."""
  shape = shape + (3,)
  if arch == 'resnet50':
    cnn = tf.keras.applications.ResNet50(
      weights='imagenet', include_top=False, input_shape=shape)
  else:
    raise ValueError('invalid arch={arch}')

  pool = layers.GlobalAveragePooling2D()
  model = tf.keras.Sequential([cnn, pool])
  for l in model.layers:
    l.trainable = trainable

  return model

def test(arch='resnet50'):
  model = load_cnn_extractor()
  model(np.zeros([1, 224, 224, 3]))
  model.summary()

if __name__ == '__main__':
  fire.Fire(test)