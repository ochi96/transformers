import tensorflow as tf
import numpy as np
from keras.layers import Layer
from keras import layer


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding_1d(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def positional_encoding_2d(row,col,d_model):
    assert d_model % 2 == 0

    row_pos = np.repeat(np.arange(row),col)[:,np.newaxis]
    col_pos = np.repeat(np.expand_dims(np.arange(col),0),row,axis=0).reshape(-1,1)
    angle_rads_row = get_angles(row_pos,np.arange(d_model//2)[np.newaxis,:],d_model//2)
    angle_rads_col = get_angles(col_pos,np.arange(d_model//2)[np.newaxis,:],d_model//2)

    angle_rads_row[:, 0::2] = np.sin(angle_rads_row[:, 0::2])
    angle_rads_row[:, 1::2] = np.cos(angle_rads_row[:, 1::2])
    angle_rads_col[:, 0::2] = np.sin(angle_rads_col[:, 0::2])
    angle_rads_col[:, 1::2] = np.cos(angle_rads_col[:, 1::2])
    pos_encoding = np.concatenate([angle_rads_row,angle_rads_col],axis=1)[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

def EncoderLayer(d_model, num_heads, dff, x, training,rate=0.1, mask=None):

    mha = tf.keras.layers.MultiHeadAttention(value_dim=d_model, key_dim=d_model, num_heads=num_heads, dropout=rate, output_shape=d_model)
    ffn = point_wise_feed_forward_network(d_model, dff)

    layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    dropout2 = tf.keras.layers.Dropout(rate)

    attn_output, _ = mha(x, x, x, mask, True, training=training)
    out1 = layernorm1(x + attn_output)

    ffn_output = ffn(out1)
    ffn_output = dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)

    return out2

def DecoderLayer(d_model, num_heads, dff, x, enc_output, training, rate=0.1, look_ahead_mask=None, padding_mask=None):
    mha1 = tf.keras.layers.MultiHeadAttention(value_dim=d_model, key_dim=d_model, num_heads=num_heads, dropout=rate, output_shape=d_model)
    mha2 = tf.keras.layers.MultiHeadAttention(value_dim=d_model, key_dim=d_model, num_heads=num_heads, dropout=rate, output_shape=d_model)

    ffn = point_wise_feed_forward_network(d_model, dff)

    layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    dropout3 = tf.keras.layers.Dropout(rate)


    attn1, attn_weights_block1 = mha1(x, x, x, look_ahead_mask, True, training=training)
    out1 = layernorm1(attn1 + x)

    attn2, attn_weights_block2 = mha2(out1, enc_output, enc_output, padding_mask,True, training=training)
    out2 = layernorm2(attn2 + out1)

    ffn_output = ffn(out2)
    ffn_output = dropout3(ffn_output, training=training)
    out3 = layernorm3(ffn_output + out2)

    return out3, attn_weights_block1, attn_weights_block2


def Encoder(self, num_layers, d_model, num_heads, dff, row_size,col_size, x, training,rate=0.1, mask=None):
    
    embedding = tf.keras.layers.Dense(d_model,activation='relu')
    pos_encoding = positional_encoding_2d(row_size,col_size,d_model)
    enc_layers = EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

    dropout = tf.keras.layers.Dropout(rate)

    seq_len = tf.shape(x)[1]
    x = embedding(x)
    x += pos_encoding[:, :seq_len, :]
    x = dropout(x, training=training)

    for i in range(num_layers):
        x = enc_layers[i](x, training, mask)

    return x

def Decoder(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding,
    x, enc_output, training,rate = 0.1 look_ahead_mask=None, padding_mask=None,):

    embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    pos_encoding = positional_encoding_1d(maximum_position_encoding, d_model)

    dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
    dropout = tf.keras.layers.Dropout(rate)

    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = embedding(x) 
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    x += pos_encoding[:, :seq_len, :]

    x = dropout(x, training=training)

    for i in range(num_layers):
        x, block1, block2 = dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

    return x, attention_weights


def Transformer(self, layers, d_model, num_heads, dff,row_size,col_size, target_vocab_size,max_pos_encoding, 
    inp, tar, training,rate=0.1, look_ahead_mask=None, dec_padding_mask=None,enc_padding_mask=None):

    encoder = Encoder(layers, d_model, num_heads, dff,row_size,col_size, rate)
    decoder = Decoder(layers, d_model, num_heads, dff, target_vocab_size,max_pos_encoding, rate)
    final_layer = tf.keras.layers.Dense(target_vocab_size)

    enc_output = encoder(inp, training, enc_padding_mask)
    dec_output, attention_weights = decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    final_output = final_layer(dec_output)

    return final_output, attention_weights
