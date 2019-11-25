""" vt.py

Video Transform.
"""

import numpy as np
import PIL
import scipy
from vidaug import augmentors as va


class Resize:
  """
  Resize video.
  Args:
    size (float, float): shato to resize.
    interp (string): Interpolation to use for re-sizing
    ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
  """

  def __init__(self, size, interp='bilinear'):
    self.size = size
    self.interpolation = interp

  def __call__(self, clip):
    if isinstance(clip[0], np.ndarray):
      return [scipy.misc.imresize(img, size=self.size,interp=self.interpolation) for img in clip]
    elif isinstance(clip[0], PIL.Image.Image):
      return [img.resize(size=self.size, resample=self._get_PIL_interp(self.interpolation)) for img in clip]
    else:
      raise TypeError('Expected numpy.ndarray or PIL.Image' +
        'but got list of {0}'.format(type(clip[0])))

  def _get_PIL_interp(self, interp):
    if interp == 'nearest':
      return PIL.Image.NEAREST
    elif interp == 'lanczos':
      return PIL.Image.LANCZOS
    elif interp == 'bilinear':
      return PIL.Image.BILINEAR
    elif interp == 'bicubic':
      return PIL.Image.BICUBIC
    elif interp == 'cubic':
      return PIL.Image.CUBIC


class VideoShapeTransform:
  """Reshape video transform."""

  def __init__(self, shape=(224, 224)):
    self.seq = va.Sequential([
      Resize(shape)
    ])

  def __call__(self, frames):
    return self.seq(frames)

class VideoTransform:
  """Video transform for data augmentation."""

  def __init__(self, shape=(224, 224)):
    sometimes = lambda aug: va.Sometimes(0.5, aug)
    self.seq = va.Sequential([
      # Add a value to all pixel intesities in an video
      sometimes(
        va.OneOf([
          va.Add(-100),
          va.Add(100),
          va.Add(-60),
          va.Add(60),
        ]),
      ),
      # Extract center crop of the video
      sometimes(
        va.OneOf([
          va.CenterCrop((224,224)),
          va.CenterCrop((200,200)),
          va.CenterCrop((180,180)),
        ]),
      ),
      # Augmenter that sets a certain fraction of pixel intesities to 255,
      # hence they become white
      sometimes(
        va.OneOf([
          va.Salt(50),
          va.Salt(100),
          va.Salt(150),
          va.Salt(200),
        ]),
      ),
      # Augmenter that sets a certain fraction of pixel intensities to 0,
      # hence they become black
      sometimes(
        va.OneOf([
          va.Pepper(50),
          va.Pepper(100),
          va.Pepper(150),
          va.Pepper(200),
        ]),
      ),
      # Rotate video randomly by a random angle within given bounds
      sometimes(
        va.OneOf([
          va.RandomRotate(degrees=5),
          va.RandomRotate(degrees=10),
          va.RandomRotate(degrees=15),
        ]),
      ),
      # Shifting video in X and Y coordinates
      sometimes(
        va.OneOf([
          va.RandomTranslate(20,20),
          va.RandomTranslate(10,10),
          va.RandomTranslate(5,5)
        ]),
      ),
      # Horizontally flip the video with 50% probability
      sometimes(va.HorizontalFlip()),
      Resize(shape)
    ])

  def __call__(self, frames):
    return self.seq(frames)
