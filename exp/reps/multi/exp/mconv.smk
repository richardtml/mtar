"""Experiment for Conv model"""

import time
from itertools import product

from tqdm import tqdm

rule rmconv:
  run:
    lr = config.get('lr', 0.01)
    epochs = config.get('epochs', 300)
    #convs_filters = [[128], [256], [256,128]]
    convs_filters = [[512], [384], [512,128], [384,128], [512,256]]
    #strategies = ('shortest', 'longest')
    strategies = ['longest']
    configs = list(product(convs_filters, strategies))
    for conv_filters, strategy in tqdm(
        configs, desc=f'EXP {rule}', ncols=75):
      conv_filters = str(conv_filters).replace(' ', '')
      print()
      shell(
        "python train.py"
        f" --exp {rule}"
        f" --hmdb51"
        f" --ucf101"
        f" --model Conv"
        f" --model_bn_in"
        f" --model_conv_filters {conv_filters}"
        f" --model_dropout 0.5"
        f" --train_strategy {strategy}"
        f" --train_epochs {epochs}"
        f" --opt_lr {lr}"
        f" --dss_sampling random"
      )
      time.sleep(1)
      print()
