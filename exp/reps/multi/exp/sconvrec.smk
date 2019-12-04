"""Experiment for Conv model"""

import time
from itertools import product

from tqdm import tqdm

rule rsconvrec:
  run:
    lr = config.get('lr', 0.01)
    epochs = config.get('epochs', 300)
    dss = ['hmdb51', 'ucf101']
    layers = [[[128], [128]], [[256], [256]]]
    rec_bis = [False, True]
    configs = list(product(dss, layers, rec_bis))
    for ds, layer, rec_bi in tqdm(
        configs, desc=f'EXP {rule}', ncols=75):
      conv_filters, rec_sizes = layer
      conv_filters = str(conv_filters).replace(' ', '')
      rec_sizes = str(rec_sizes).replace(' ', '')
      print()
      shell(
        "python train.py"
        f" --exp {rule}"
        f" --{ds}"
        f" --model ConvRec"
        f" --model_conv_filters {conv_filters}"
        f" --model_rec_sizes {rec_sizes}"
        f" --model_rec_bi {rec_bi}"
        f" --model_dropout 0.5"
        f" --train_epochs {epochs}"
        f" --opt_lr {lr}"
        f" --dss_sampling random"
      )
      time.sleep(1)
      print()
