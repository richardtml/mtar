"""Experiment for single frame model"""

from itertools import product

rule sframe:
  run:
    lr = config.get('lr', 1e-3)
    epochs = config.get('epochs', 500)
    dss = ('hmdb51', 'ucf101')
    ifcs = (False, True)
    configs = (dss, ifcs)
    for ds, ifc in product(*configs):
      shell(
        "python train.py"
        f" --exp_name sframe"
        f" --ds {ds}"
        f" --model SFrame"
        f" --ifc {ifc}"
        f" --dropout 0.5"
        f" --lr {lr}"
        f" --epochs {epochs}"
      )
