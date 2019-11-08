"""Experiment for single frame model"""

from itertools import product

rule srec:
  run:
    lr = config.get('lr', 1e-3)
    epochs = config.get('epochs', 500)
    dss = ('hmdb51', 'ucf101')
    rec_types = ('gru, lstm')
    rec_layerss = (1, 2, 4)
    configs = (dss, rec_types, rec_layerss)
    for ds, rec_type, rec_layers in product(*configs):
      shell(
        "python train.py"
        f" --exp_name {rule}"
        f" --ds {ds}"
        f" --model Rec"
        f" --rec_type {rec_type}"
        f" --rec_layers {rec_layers}"
        f" --lr {lr}"
        f" --epochs {epochs}"
      )
