"""Experiment for Conv model"""

import time
from itertools import product

from tqdm import tqdm

rule rsconv:
  run:
    lr = config.get('lr', 0.01)
    epochs = config.get('epochs', 250)
    dss = ['hmdb51', 'ucf101']
    samplings = ['fixed', 'random']
    convs_filters = [[512], [256], [128], [256,128], [512,256,128]]
    configs = list(product(dss, convs_filters, samplings))
    for ds, conv_filters, sampling in tqdm(
        configs, desc=f'EXP {rule}', ncols=75):
      conv_filters = str(conv_filters).replace(' ', '')
      print()
      shell(
        "python train.py"
        f" --exp {rule}"
        f" --{ds}"
        f" --model Conv"
        f" --model_conv_filters {conv_filters}"
        f" --model_dropout 0.5"
        f" --train_epochs {epochs}"
        f" --opt_lr {lr}"
        f" --dss_sampling {sampling}"
      )
      time.sleep(1)
      print()
