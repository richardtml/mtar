# Seminar experiment

from itertools import product


rule seminar:
  """Seminar 22/10/19"""
  run:
    epochs = 500
    dss = ('hmdb51', 'ucf101')
    models = ('Conv2D', 'Conv2D1D', 'FullConv')
    filters_sizes = (128, 256)
    dropout = 0
    configs = (dss, models, filters_sizes)
    for ds, model, conv2d_filters in product(*configs):
      shell(
        "python train.py"
        f" --ds {ds}"
        f" --model {model}"
        f" --conv2d_filters {conv2d_filters}"
        f" --dropout 0"
        f" --epochs {epochs}"
      )
    for ds, model, conv2d_filters in product(*configs):
      shell(
        "python train.py"
        f" --ds {ds}"
        f" --model {model}"
        f" --conv2d_filters {conv2d_filters}"
        f" --dropout 0.5"
        f" --lr 1e-2"
        f" --epochs {epochs}"
      )
    rec_sizes = (128, 256)
    configs = (dss, rec_sizes)
    for ds, rec_size in product(*configs):
      shell(
        "python train.py"
        f" --ds {ds}"
        f" --model Rec"
        f" --rec_size {rec_size}"
        f" --epochs {epochs}"
      )
