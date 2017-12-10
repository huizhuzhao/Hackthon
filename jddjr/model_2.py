#!/usr/bin/env python
# encoding: utf-8
# Created Time: 日 12/10 00:02:09 2017

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from Hackthon.jddjr import utils


def build_model(input_seq_len, output_seq_len):

    RNN = layers.LSTM
    encoder_layers = 1
    decoder_layers = 1
    hidden_dim = 100
    model = models.Sequential()

    model.add(layers.TimeDistributed(layers.Dense(100, activation='relu'), input_shape=(input_seq_len, 1)))

    for _ in range(encoder_layers):
        model.add(RNN(hidden_dim, return_sequences=True))
    model.add(RNN(hidden_dim, return_sequences=False))

    model.add(layers.RepeatVector(output_seq_len))
    for _ in range(decoder_layers):
        model.add(RNN(hidden_dim, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(1)))

    optimizer = optimizers.Adam(lr=0.01)
    def score_func(y_true, y_pred):
        y_true = tf.reduce_sum(y_true, axis=1)
        y_pred = tf.reduce_sum(y_pred, axis=1)
        mae = tf.reduce_sum(tf.abs(y_true - y_pred))
        score = mae / tf.reduce_sum(y_true)
        return score
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', score_func])

    print('model input shape: {0}'.format(model.input_shape))
    print('model output shape: {0}'.format(model.output_shape))
    return model




def main():
    input_seq_len = 90
    output_seq_len = 30
    batch_size = 64
    epochs = 20
    sale_amt = utils.get_sale_amt_by_day(range(1, 300))
    datat = utils.get_seq2seq_data(sale_amt, input_seq_len, output_seq_len)

    model = build_model(input_seq_len, output_seq_len)

    log_dir = os.path.join(utils.DATA_DIR, 'logs_seq2seq')
    callbacks = utils.get_callbacks(log_dir)

    print(model.summary())
    model.fit(datat.train_X, datat.train_Y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)




if __name__ == '__main__':
    main()
