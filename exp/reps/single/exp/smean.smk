"""Experiment for mean temporal fusion"""

from itertools import product

rule smean:
  run:
    lr = config.get('lr', 1e-3)
    epochs = config.get('epochs', 500)
    dss = ('hmdb51', 'ucf101')
    models = ('FCMean', 'MeanFC')
    ifcs = (False, True)
    configs = (dss, models, ifcs)
    for ds, model, ifc in product(*configs):
      shell(
        "python train.py"
        f" --exp_name {rule}"
        f" --ds {ds}"
        f" --model {model}"
        f" --ifc {ifc}"
        f" --dropout 0.5"
        f" --lr {lr}"
        f" --epochs {epochs}"
      )
