"""Experiment for single frame model"""

from itertools import product

rule mconv2d1d:
  run:
    lr = config.get('lr', 1e-2)
    epochs = config.get('epochs', 500)
    bns = ((0, 0), (1, 0), (1, 1))
    convs = (160, 192, 224)
    dropouts = (0.0, 0.5)
    strategies = ('shortest', 'longest', 'refill', 'interleave')
    configs = (bns, convs, convs, dropouts, strategies)
    for (bn_in, bn_out), conv2d, conv1d, dropout, strategy in product(*configs):
      cmd = (
        "python train.py"
        f" --exp {rule}"
        f" --model Conv2D1D"
        f" --model_bn_in {bn_in}"
        f" --model_bn_out {bn_out}"
        f" --model_conv2d_filters {conv2d}"
        f" --model_conv1d_filters {conv1d}"
        f" --model_dropout {dropout}"
        f" --train_strategy {strategy}"
        f" --train_epochs {epochs}"
        f" --opt_lr {lr}"
      )
      shell(cmd)
