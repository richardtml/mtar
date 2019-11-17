"""Experiment for single frame model"""

from itertools import product

rule msframe:
  run:
    lr = config.get('lr', 1e-2)
    epochs = config.get('epochs', 500)
    bns = ((0, 0), (1, 0), (1, 1))
    strategies = ('shortest', 'longest', 'refill', 'interleave')
    configs = (bns, strategies)
    for (bn_in, bn_out), strategy in product(*configs):
      cmd = (
        "python train.py"
        f" --exp {rule}"
        f" --model SFrame"
        f" --model_bn_in {bn_in}"
        f" --model_bn_out {bn_out}"
        f" --train_strategy {strategy}"
        f" --train_epochs {epochs}"
        f" --opt_lr {lr}"
      )
      shell(cmd)
