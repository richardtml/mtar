"""Experiment for Rec model"""

import time
from itertools import product

from tqdm import tqdm

rule rmrec:
  run:
    lr = config.get('lr', 0.01)
    rec_bi = config.get('model_rec_bi', False)
    epochs = config.get('epochs', 300)
    dss_cache = config.get('dss_cache', False)
    recs_sizes = [[512], [256,128], [512,512]]
    strategies = ['shortest', 'longest']
    configs = list(product(recs_sizes, strategies))
    for rec_sizes, strategy in tqdm(configs,
        desc=f'EXP {rule}', ncols=75):
      rec_sizes = str(rec_sizes).replace(' ', '')
      print()
      shell(
        "python train.py"
        f" --exp {rule}"
        f" --hmdb51"
        f" --ucf101"
        f" --model Rec"
        f" --model_bn_in"
        f" --model_rec_sizes {rec_sizes}"
        f" --model_rec_bi {rec_bi}"
        f" --train_strategy {strategy}"
        f" --train_epochs {epochs}"
        f" --opt_lr {lr}"
        f" --dss_sampling random"
        f" --dss_cache {dss_cache}"
      )
      time.sleep(1)
      print()