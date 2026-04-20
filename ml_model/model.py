"""
==============================================================================
  STGCN-LSTM Model Architecture (TensorFlow/Keras)
==============================================================================

Ported from STGCN-rehab-main/demo.py.
Builds the Spatial-Temporal Graph Convolutional Network with LSTM head
for rehabilitation exercise quality assessment.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    concatenate, Flatten, Dropout, Dense, Input, LSTM, ConvLSTM2D
)
from tensorflow.keras.models import Model


def build_sgcn_block(inp, bias_mat_1, bias_mat_2):
    """Single SGCN block matching the architecture in GCN/sgcn_lstm.py"""
    # Temporal convolution
    k1 = tf.keras.layers.Conv2D(64, (9, 1), padding='same', activation='relu')(inp)
    k = concatenate([inp, k1], axis=-1)

    # First hop
    x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, activation='relu')(k)
    expand_x1 = tf.expand_dims(x1, axis=3)
    f_1 = ConvLSTM2D(filters=25, kernel_size=(1, 1), return_sequences=True)(expand_x1)
    f_1 = f_1[:, :, :, 0, :]
    coefs_1 = tf.nn.softmax(tf.nn.leaky_relu(f_1) + bias_mat_1)
    gcn_x1 = tf.keras.layers.Lambda(
        lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1])
    )([coefs_1, x1])

    # Second hop
    y1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, activation='relu')(k)
    expand_y1 = tf.expand_dims(y1, axis=3)
    f_2 = ConvLSTM2D(filters=25, kernel_size=(1, 1), return_sequences=True)(expand_y1)
    f_2 = f_2[:, :, :, 0, :]
    coefs_2 = tf.nn.softmax(tf.nn.leaky_relu(f_2) + bias_mat_2)
    gcn_y1 = tf.keras.layers.Lambda(
        lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1])
    )([coefs_2, y1])

    gcn_1 = concatenate([gcn_x1, gcn_y1], axis=-1)

    # Temporal convolution
    z1 = tf.keras.layers.Conv2D(16, (9, 1), padding='same', activation='relu')(gcn_1)
    z1 = Dropout(0.25)(z1)
    z2 = tf.keras.layers.Conv2D(16, (15, 1), padding='same', activation='relu')(z1)
    z2 = Dropout(0.25)(z2)
    z3 = tf.keras.layers.Conv2D(16, (20, 1), padding='same', activation='relu')(z2)
    z3 = Dropout(0.25)(z3)
    z = concatenate([z1, z2, z3], axis=-1)
    return z


def build_model(num_joints, num_channel, bias_mat_1, bias_mat_2):
    """Rebuild the full STGCN-LSTM model from code."""
    seq_input = Input(shape=(None, num_joints, num_channel), batch_size=None)
    x = build_sgcn_block(seq_input, bias_mat_1, bias_mat_2)
    y = build_sgcn_block(x, bias_mat_1, bias_mat_2)
    y = y + x
    z = build_sgcn_block(y, bias_mat_1, bias_mat_2)
    z = z + y

    # LSTM head
    reshaped = tf.keras.layers.Reshape(target_shape=(-1, z.shape[2] * z.shape[3]))(z)
    rec = LSTM(80, return_sequences=True)(reshaped)
    rec = Dropout(0.25)(rec)
    rec1 = LSTM(40, return_sequences=True)(rec)
    rec1 = Dropout(0.25)(rec1)
    rec2 = LSTM(40, return_sequences=True)(rec1)
    rec2 = Dropout(0.25)(rec2)
    rec3 = LSTM(80)(rec2)
    rec3 = Dropout(0.25)(rec3)
    out = Dense(1, activation='linear')(rec3)

    model = Model(seq_input, out)
    return model
