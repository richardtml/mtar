"""Test number of filters for conv2d and conv1d archs"""

from itertools import product

rule conv_filters:
  run:
    epochs = config["epochs"] if "epochs" in config else 500
    dss = ('hmdb51', 'ucf101')
    models = ('Conv2D', 'Conv2D1D', 'FullConv')
    conv_filters = (96, 128, 160)
    configs = (dss, models, conv_filters, conv_filters)
    for ds, model, conv2d_filters, conv1d_filters in product(*configs):
      shell(
        "python train.py"
        f" --exp_name {rule}"
        f" --ds {ds}"
        f" --model {model}"
        f" --conv2d_filters {conv2d_filters}"
        f" --conv1d_filters {conv1d_filters}"
        f" --dropout 0.5"
        f" --epochs {epochs}"
      )
