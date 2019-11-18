"""Experiment for single frame model"""

from itertools import product

rule mmean:
  run:
    lr = config.get('lr', 1e-2)
    epochs = config.get('epochs', 500)
    models = ('FCMean', 'MeanFC')
    bns_in = (0, 1)
    strategies = ('shortest', 'longest', 'refill', 'interleave')
    configs = (models, bns_in, strategies)
    for model, bn_in, strategy in product(*configs):
      cmd = (
        "python train.py"
        f" --exp {rule}"
        f" --model {model}"
        f" --model_bn_in {bn_in}"
        f" --train_strategy {strategy}"
        f" --train_epochs {epochs}"
        f" --opt_lr {lr}"
      )
      shell(cmd)
