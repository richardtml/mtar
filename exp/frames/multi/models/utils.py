""" utils.py

Models utils.
"""

import sys
from exp.frames.multi.models import models

import tensorflow as tf
from tensorflow.keras import layers


def get_model_class(class_name):
  try:
    return getattr(models, class_name)
  except AttributeError as ex:
    raise AttributeError(f'model class "{class_name}" does not exist') from ex


def load_cnn_extractor(arch='resnet50', shape=(224, 224)):
  """Loads CNN model pretrained with Imagenet without top layers."""
  if arch == 'resnet50':
    cnn = tf.keras.applications.ResNet50(
      weights='imagenet', include_top=False, input_shape=shape)
  else:
    raise ValueError('invalid arch={arch}')

  pool = layers.GlobalAveragePooling2D()
  model = tf.keras.Sequential([cnn, pool])

  return model

  def test_cnn(arch='resnet50'):
    cnn = load_cnn_extractor()