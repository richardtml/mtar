"""Experiment for mean Temporal fusion"""

from itertools import product

rule smean:
  run:
    epochs = config["epochs"] if "epochs" in config else 500
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
        f" --epochs {epochs}"
      )
