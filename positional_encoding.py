import tensorflow as tf
import numpy as np

# Maybe utilize https://github.com/tensorflow/models/tree/master/official/nlp/keras_nlp instead of our own positional encoding implementation

class PositionalEncoding(tf.keras.layers.Layer):
  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)
    # apply sin to even index in the array
    sines = tf.math.sin(angle_rads[:, 0::2])
    # apply cos to odd index in the array
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class PositionalEncoding2D(tf.keras.layers.Layer):

  def __init__(self, d_model):
    super(PositionalEncoding2D, self).__init__()
    self.pos_encoding = self.positional_encoding(d_model)

  def get_angles(self, position, i, d_model):
    # angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    # return position * angles
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return position * angle_rates

  def positional_encoding(self, d_model):
      assert d_model % 2 == 0

      row_pos = np.repeat(np.arange(9), 9)[:, np.newaxis]
      col_pos = np.repeat(np.expand_dims(np.arange(9), 0), 9, axis=0).reshape(-1, 1)
      angle_rads_row = self.get_angles(row_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2)
      angle_rads_col = self.get_angles(col_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2)

      angle_rads_row[:, 0::2] = tf.math.sin(angle_rads_row[:, 0::2])
      angle_rads_row[:, 1::2] = tf.math.cos(angle_rads_row[:, 1::2])
      angle_rads_col[:, 0::2] = tf.math.sin(angle_rads_col[:, 0::2])
      angle_rads_col[:, 1::2] = tf.math.cos(angle_rads_col[:, 1::2])
      pos_encoding = [[angle_rads_row, angle_rads_col]][np.newaxis, ...]

      return tf.cast(pos_encoding, dtype=tf.float32)

  def call(self, inputs, initial_inputs):
    return inputs + self.pos_encoding[:, :tf.shape(initial_inputs)[1], :]