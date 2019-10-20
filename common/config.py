""" config.py

Config utils.
"""

import os
from dotenv import load_dotenv


VARS = {}


def load_var(var):
  if var not in os.environ:
    load_dotenv()
  if var not in os.environ:
    raise ValueError(
        f'{var} environment variable not defined, see README.md'
    )


def load():
  load_var('DATASETS_DIR')
  VARS['DATASETS_DIR'] = os.getenv('DATASETS_DIR')
  load_var('RESULTS_DIR')
  VARS['RESULTS_DIR'] = os.getenv('RESULTS_DIR')
  load_var('SEED')
  VARS['SEED'] = int(os.getenv('SEED'))


def get(var):
  return VARS[var]


load()
