"""Experiment for single frame model"""

from itertools import product

rule msframe:
  run:
    lr = config.get('lr', 1e-2)
    epochs = config.get('epochs', 500)
    bns = ((0, 0, 0), (1, 0, 0), (1, 1, 1))
    strategies = ('shortest', 'longest', 'refill', 'interleave')
    configs = (bns, strategies)
    for bn, strategy in product(*configs):
      bn_in, hmdb51_bn_out, ucf101_bn_out = bn
      model = {
        'name': 'SFrame',
        'bn_in': bn_in
      }
      tasks={
        'hmdb51': {'split': 1, 'bn_out': hmdb51_bn_out},
        'ucf101': {'split': 1, 'bn_out': ucf101_bn_out},
      }
      train={
        'strategy': strategy,
        'epochs': epochs,
        'lr': lr,
        'optimizer': 'sgd',
        'momentum': 0.0,
        'nesterov': False,
        'tbatch': 128,
        'ebatch': 64,
        'alphas': [1, 1],
      }
      shell(
        "python train.py"
        f" --exp {rule}"
        f" --model \"{model}\""
        f" --tasks \"{tasks}\""
        f" --train \"{train}\""
      )
