"""Experiment for Rec model"""

import time
from itertools import product

from tqdm import tqdm

rule rsrec:
  run:
    lr = config.get('lr', 0.01)
    epochs = config.get('epochs', 250)
    dss = ['hmdb51', 'ucf101']
    samplings = ['fixed', 'random']
    recs_sizes = [[512], [256], [128], [256,128], [512,512]]
    configs = [dss, recs_sizes, samplings]
    configs = list(product(dss, recs_sizes, samplings))
    for ds, rec_sizes, sampling in tqdm(configs,
        desc=f'EXP {rule}', ncols=75):
      rec_sizes = str(rec_sizes).replace(' ', '')
      print()
      shell(
        "python train.py"
        f" --exp {rule}"
        f" --{ds}"
        f" --model Rec"
        f" --model_rec_sizes {rec_sizes}"
        f" --train_epochs {epochs}"
        f" --opt_lr {lr}"
        f" --dss_sampling {sampling}"
      )
      time.sleep(1)
      print()