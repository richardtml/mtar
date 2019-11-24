""" vt.py

Video Transform.
"""

from vidaug import augmentors as va


class VideoShapeTransform:
  """Video transform for data augmentation."""

  def __init__(self, shape=(224, 224)):
    self.shape = shape

  def __call__(self, frames):
    frames = [frame.resize(self.shape) for frame in frames]
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
          va.RandomRotate(degrees=10),
          va.RandomRotate(degrees=15),
          va.RandomRotate(degrees=25),
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
      sometimes(va.HorizontalFlip())
    ])
    self.shape = shape

  def __call__(self, frames):
    frames = self.seq(frames)
    frames = [frame.resize(self.shape) for frame in frames]
    return frames
