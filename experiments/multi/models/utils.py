""" utils.py

Models utils.
"""

import sys
from experiments.multi.models import models


def get_model_class(class_name):
  try:
    return getattr(models, class_name)
  except AttributeError as ex:
    raise AttributeError(f'model class "{class_name}" does not exist') from ex
