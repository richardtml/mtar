""" config.py

Config utils.
"""

import os
from dotenv import load_dotenv

VARS = {}
DEFAULT_SEED = 0

def load_var(key, val=None, convert=None):
  if key in os.environ:
    val = os.getenv(key)
    VARS[key] = convert(val) if convert else val
  else:
    if val is not None:
      VARS[key] = convert(val) if convert else val
    else:
      raise ValueError(
        f'{key} environment variable not defined, see README.md')

def load():
  load_dotenv()
  load_var('DATASETS_DIR')
  load_var('RESULTS_DIR')
  load_var('SEED', val=DEFAULT_SEED, convert=int)

def get(var):
  if len(VARS) == 0: load()
  return VARS[var]
