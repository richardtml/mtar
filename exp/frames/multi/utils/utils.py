""" utils.py
"""

from collections import namedtuple

def build_datasets_cfg(cfg, dss_names=('hmdb51', 'ucf101')):
  dss = []
  DS = namedtuple('DS', ('name', 'split'))
  for name in dss_names:
    enabled = getattr(cfg, name, False)
    if enabled:
      split = getattr(cfg, 'split', 1)
      dss.append(DS(name, split))
  return dss
