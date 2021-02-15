import tensorflow as tf
import numpy as np
from positional_encoding import PositionalEncoding2D, PositionalEncoding
from masking import create_padding_mask, create_look_ahead_mask

def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input((None, d_model), name="inputs")
    attention = tf.keras.layers.MultiHeadAttention(value_dim=d_model, key_dim=d_model, num_heads=num_heads, dropout=0.1, output_shape=d_model)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    outputs = tf.keras.layers.Dense(units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(d_model)(outputs)
    outputs = tf.keras.layers.Dropout(dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs], outputs=outputs, name=name)


def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
    inputs = tf.keras.Input((None,), name="inputs")
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings = PositionalEncoding2D(d_model)(embeddings, inputs)
    outputs = tf.keras.layers.Dropout(dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer_{}".format(i), )([outputs])

    return tf.keras.Model(inputs=[inputs], outputs=outputs, name=name)


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input((None, d_model), name="inputs")
    enc_outputs = tf.keras.Input((None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input((1, None, None), name="look_ahead_mask")

    attention1 = tf.keras.layers.MultiHeadAttention(num_heads, d_model, d_model, dropout, output_shape=d_model)(inputs,inputs,inputs,look_ahead_mask)
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)
    attention2 = tf.keras.layers.MultiHeadAttention(num_heads, d_model, d_model, dropout, output_shape=d_model)(attention1, enc_outputs, enc_outputs)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + attention2)
    outputs = tf.keras.layers.Dense(units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(d_model)(outputs)
    outputs = tf.keras.layers.Dropout(dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask], outputs=outputs, name=name)


def decoder(vocab_size, num_layers, units, d_model, num_heads, max_pos_encoding, dropout, name='decoder'):
    inputs = tf.keras.Input((None,), name='inputs')
    enc_outputs = tf.keras.Input((None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input((1, None, None), name='look_ahead_mask')
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(max_pos_encoding, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(units, d_model=d_model, num_heads=num_heads, dropout=dropout, name='decoder_layer_{}'.format(i), )(inputs=[outputs, enc_outputs, look_ahead_mask])

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask], outputs=outputs, name=name)


def transformer(vocab_size, num_layers, units, d_model, num_heads, dropout, max_pos_encoding, name="transformer"):
    inputs = tf.keras.Input((None,), name="inputs")
    dec_inputs = tf.keras.Input((None,), name="dec_inputs")
    look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(inputs)
    dec_padding_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, 1, None), name='dec_padding_mask')(dec_inputs)
    enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, units=units, d_model = d_model, num_heads=num_heads, dropout=dropout, )(inputs=[inputs])
    dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, units=units, d_model = d_model, num_heads=num_heads, max_pos_encoding=max_pos_encoding, dropout=dropout, )(
        inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
    outputs = tf.keras.layers.Dense(vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[(inputs, dec_inputs)], outputs=outputs, name=name)