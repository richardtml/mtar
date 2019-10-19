""" utils.py

Common utils.
"""

import datetime
import time


def timestamp(fmt='%y%m%dT%H%M%S'):
  """Returns current timestamp."""
  return datetime.datetime.fromtimestamp(time.time()).strftime(fmt)
