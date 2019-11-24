""" vt.py

Video Transform.
"""

from vidaug import augmentors as va

class VideoTransform:
  """Video transform for data augmentation."""

  def __init__(self, shape=(224, 224)):
    sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
    self.seq = va.Sequential([
      va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
      va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]
      sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
    ])
    self.shape = shape

  def __call__(self, frames):
    frames = self.seq(frames)
    frames = [frame.resize(self.shape) for frame in frames]
    return frames
