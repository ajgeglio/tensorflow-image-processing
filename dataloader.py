import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import os

class DataLoader:
  def __init__(self, img_size = (1306, 2458), num_channels=3):
    self.img_size = img_size
    self.num_channels = num_channels

  @tf.function
  def load_image(self, file_name):
    raw = tf.io.read_file(file_name)
    tensor = tf.io.decode_image(raw)
    tensor = tf.keras.preprocessing.image.smart_resize(tensor, self.img_size)
    tensor = tf.cast(tensor, tf.float32)
    return tensor
  
  def create_dataset(self, file_names, labels):
    with tf.device('/cpu:0'):
      dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))
      dataset = dataset.map(lambda file_name, label: (self.load_image(file_name), label))
    return dataset
  
  # @tf.function
  # def load_image_nl(self, file_name):
  #   raw = tf.io.read_file(file_name)
  #   tensor = tf.io.decode_image(raw, channels=self.num_channels, expand_animations=False)
  #   tensor.set_shape((self.img_size[0], self.img_size[1], self.num_channels))
  #   tensor = tf.image.resize(tensor, self.img_size, method = 'bilinear')
  #   return tensor

  @tf.function
  def load_image_nl(self, file_path_tensor):
      """Loads and processes an image for TensorFlow inference without labels."""
      ext = tf.strings.split(file_path_tensor, sep=".")[-1] # Get the file extension
      ext = tf.strings.lower(ext) # Convert the extension to lowercase
      raw = tf.io.read_file(file_path_tensor) # Read the raw file
      def decode_png_jpg():
          return tf.io.decode_image(raw, channels=0) # Decode PNG and JPG images, 
          # channels=0 allows TensorFlow to dynamically adjust based on the actual 
          # image data, making it flexible for different image types
      def decode_tiff():
          return tfio.experimental.image.decode_tiff(raw) # Decode TIFF images
      tensor = tf.cond(
          tf.reduce_any(tf.equal(ext, ["png", "jpg"])), # Check if the file is PNG or JPG
          decode_png_jpg, # Decode PNG or JPG
          decode_tiff # Otherwise, decode TIFF
      )
      tensor = tf.cond(
          tf.equal(tf.shape(tensor)[-1], 4),
          lambda: tensor[..., :3], # Remove the alpha channel if present
          lambda: tensor
      )
      if tf.equal(tf.shape(tensor)[-1], 1):
          tensor = tf.image.grayscale_to_rgb(tensor) # Convert grayscale to RGB
      tensor.set_shape((self.img_size[0], self.img_size[1], 3)) # Set the shape
      tensor = tf.image.resize(tensor, self.img_size, method='bilinear') # Resize the image
      return tensor

  def create_dataset_nl(self, file_names):
    with tf.device('/cpu:0'):
      dataset = tf.data.Dataset.from_tensor_slices((file_names))
      dataset = dataset.map(lambda file_path_tensor: self.load_image_nl(file_path_tensor))
      dataset = dataset.batch(32)
    return dataset

