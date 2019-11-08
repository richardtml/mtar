"""Study convolutional architectures"""

from itertools import product

rule sconv:
  run:
    epochs = config.get('epochs', 500)
    dss = ('hmdb51', 'ucf101')
    models = ('Conv2D', 'Conv2D1D', 'FullConv')
    filters = (96, 128, 160, 192)
    dropouts = (0.0, 0.5)
    configs = (dss, models, filters, filters, dropouts)
    for ds, model, conv2d, conv1d, do in product(*configs):
      shell(
        "python train.py"
        f" --exp_name {rule}"
        f" --ds {ds}"
        f" --model {model}"
        f" --conv2d_filters {conv2d}"
        f" --conv1d_filters {conv1d}"
        f" --dropout {do}"
        f" --epochs {epochs}"
      )
