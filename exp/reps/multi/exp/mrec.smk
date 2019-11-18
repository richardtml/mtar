"""Experiment for single frame model"""

from itertools import product

rule mrec:
  run:
    lr = config.get('lr', 1e-2)
    epochs = config.get('epochs', 500)
    bns = ((0, 0), (1, 0), (1, 1))
    rec_sizes = (128, 160)
    strategies = ('shortest', 'longest', 'refill', 'interleave')
    configs = (bns, rec_sizes, strategies)
    for (bn_in, bn_out), rec_size, strategy in product(*configs):
      cmd = (
        "python train.py"
        f" --exp {rule}"
        f" --model Rec"
        f" --model_bn_in {bn_in}"
        f" --model_bn_out {bn_out}"
        f" --model_rec_type gru"
        f" --model_rec_size {rec_size}"
        f" --train_strategy {strategy}"
        f" --train_epochs {epochs}"
        f" --opt_lr {lr}"
      )
      shell(cmd)
