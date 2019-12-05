"""Experiment for Rec model"""

import time
from itertools import product

from tqdm import tqdm

rule rmconvrec:
  run:
    lr = config.get('lr', 0.01)
    rec_bi = config.get('model_rec_bi', False)
    epochs = config.get('epochs', 300)
    dss_cache = config.get('dss_cache', False)
    layers = [[[512], [512]], [[384], [384]], [[256], [256]], [[128], [128]]]
    strategies = ['longest']
    configs = list(product(layers, strategies))
    for layer, strategy in tqdm(configs,
        desc=f'EXP {rule}', ncols=75):
      conv_filters, rec_sizes = layer
      conv_filters = str(conv_filters).replace(' ', '')
      rec_sizes = str(rec_sizes).replace(' ', '')
      print()
      shell(
        "python train.py"
        f" --exp {rule}"
        f" --hmdb51"
        f" --ucf101"
        f" --model ConvRec"
        f" --model_bn_in"
        f" --model_conv_filters {conv_filters}"
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