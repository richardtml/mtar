""" config.py

Config utils.
"""

import os
from dotenv import load_dotenv


def load_var(var):
  if var not in os.environ:
    load_dotenv()
  if var not in os.environ:
    raise ValueError(
        f'{var} environment variable not defined, see README.md'
    )


def load():
  load_var('DATASETS_DIR')
  load_var('RESULTS_DIR')
  load_var('SEED')


def get(var):
  return os.getenv(var)


load()
