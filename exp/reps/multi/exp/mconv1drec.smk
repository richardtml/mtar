"""Experiment for single frame model"""

from itertools import product

rule mconv1drec:
  run:
    lr = config.get('lr', 5e-3)
    epochs = config.get('epochs', 500)
    strategies = ('shortest', 'longest')
    rec_layerss = (1, 2)
    rec_bi_merges = ('concat', 'ave')
    configs = (rec_layerss, rec_bi_merges, strategies)
    for rec_layers, merge, strategy in product(*configs):
      shell(
        "python train.py"
        f" --exp {rule}"
        f" --model Conv1DRec"
        f" --model_bn_in 1"
        f" --model_conv1d_filters 256"
        f" --model_rec_size 256"
        f" --model_rec_layers {rec_layers}"
        f" --model_rec_bi 1"
        f" --model_rec_bi_merge {merge}"
        f" --model_dropout 0.5"
        f" --train_strategy {strategy}"
        f" --train_epochs {epochs}"
        f" --opt_lr {lr}"
        f" --dss_sampling random"
      )
    configs = (rec_layerss, strategies)
    for rec_layers, strategy in product(*configs):
      shell(
        "python train.py"
        f" --exp {rule}"
        f" --model Conv1DRec"
        f" --model_bn_in 1"
        f" --model_conv1d_filters 256"
        f" --model_rec_size 256"
        f" --model_rec_layers {rec_layers}"
        f" --model_dropout 0.5"
        f" --train_strategy {strategy}"
        f" --train_epochs {epochs}"
        f" --opt_lr {lr}"
        f" --dss_sampling random"
      )
