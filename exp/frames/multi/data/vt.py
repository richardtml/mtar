""" vt.py

Video Transform.
"""

import numbers
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

class CenterCrop(object):
  """
  Extract center crop of thevideo.
  Args:
    size (sequence or int): Desired output size for the crop in format (h, w).
  """
  def __init__(self, size):
    if isinstance(size, numbers.Number):
      if size < 0:
        raise ValueError('If size is a single number, it must be positive')
      size = (size, size)
    else:
      if len(size) != 2:
        raise ValueError('If size is a sequence, it must be of len 2.')
    self.size = size

  def __call__(self, clip):
    crop_h, crop_w = self.size
    if isinstance(clip[0], np.ndarray):
      im_h, im_w, _ = clip[0].shape
    elif isinstance(clip[0], PIL.Image.Image):
      im_w, im_h = clip[0].size
    else:
      raise TypeError('Expected numpy.ndarray or PIL.Image' +
        'but got list of {0}'.format(type(clip[0])))

    if crop_w > im_w or crop_h > im_h:
      return clip
      # error_msg = ('Initial image size should be larger then' +
      #   'cropped size but got cropped sizes : ' +
      #   '({w}, {h}) while initial image is ({im_w}, ' +
      #   '{im_h})'.format(im_w=im_w, im_h=im_h, w=crop_w, h=crop_h))
      # raise ValueError(error_msg)

    w1 = int(round((im_w - crop_w) / 2.))
    h1 = int(round((im_h - crop_h) / 2.))

    if isinstance(clip[0], np.ndarray):
      return [img[h1:h1 + crop_h, w1:w1 + crop_w, :] for img in clip]
    elif isinstance(clip[0], PIL.Image.Image):
      return [img.crop((w1, h1, w1 + crop_w, h1 + crop_h)) for img in clip]

class VideoShapeTransform:
  """Reshape video transform."""

  def __init__(self, shape=(224, 224)):
    self.seq = va.Sequential([
      Resize(shape)
    ])

  def __call__(self, frames):
    frames = [frame for frame in self.seq(frames)]
    frames = np.stack([np.asarray(frame) for frame in frames])
    frames = np.divide(frames, 255, dtype=np.float32)
    return frames

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
          CenterCrop((224,224)),
          CenterCrop((200,200)),
          CenterCrop((180,180)),
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
    frames = [frame for frame in self.seq(frames)]
    frames = np.stack([np.asarray(frame) for frame in frames])
    frames = np.divide(frames, 255, dtype=np.float32)
    return frames
