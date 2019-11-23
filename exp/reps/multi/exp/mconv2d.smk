"""Experiment for single frame model"""

from itertools import product

rule mconv2d:
  run:
    lr = config.get('lr', 1e-2)
    epochs = config.get('epochs', 500)
    bns = ((0, 0), (1, 0), (1, 1))
    conv2ds = (160, 192, 224)
    dropouts = (0.0, 0.5)
    strategies = ('shortest', 'longest', 'refill', 'interleave')
    configs = (bns, conv2ds, dropouts, strategies)
    for (bn_in, bn_out), conv2d, dropout, strategy in product(*configs):
      cmd = (
        "python train.py"
        f" --exp {rule}"
        f" --model Conv2D"
        f" --model_bn_in {bn_in}"
        f" --model_bn_out {bn_out}"
        f" --model_conv2d_filters {conv2d}"
        f" --model_dropout {dropout}"
        f" --train_strategy {strategy}"
        f" --train_epochs {epochs}"
        f" --opt_lr {lr}"
      )
      shell(cmd)

rule mconv2d_tune:
  run:
    lr = config.get('lr', 5e-3)
    epochs = config.get('epochs', 500)
    conv2ds = (320, 352)
    strategies = ('shortest', 'refill', 'longest', 'interleave')
    configs = (conv2ds, strategies)
    for conv2d, strategy in product(*configs):
      cmd = (
        "python train.py"
        f" --exp {rule}"
        f" --dss_cache 1"
        f" --model Conv2D"
        f" --model_bn_in {1}"
        f" --model_bn_out {0}"
        f" --model_conv2d_filters {conv2d}"
        f" --model_dropout {0.5}"
        f" --model_ifc 0"
        f" --train_strategy {strategy}"
        f" --train_epochs {epochs}"
        f" --opt_lr {lr}"
      )
      shell(cmd)


rule mconv2d_sampling:
  run:
    lr = config.get('lr', 5e-3)
    epochs = config.get('epochs', 500)
    conv2ds = (224, 256)
    strategies = ('shortest', 'refill', 'longest', 'interleave')
    configs = (conv2ds, strategies)
    for conv2d, strategy in product(*configs):
      cmd = (
        "python train.py"
        f" --exp {rule}"
        f" --dss_cache 1"
        f" --model Conv2D"
        f" --model_bn_in {1}"
        f" --model_bn_out {0}"
        f" --model_conv2d_filters {conv2d}"
        f" --model_dropout {0.5}"
        f" --model_ifc 0"
        f" --train_strategy {strategy}"
        f" --train_epochs {epochs}"
        f" --opt_lr {lr}"
        f" --dss_sampling random"
      )
      shell(cmd)