#!/usr/bin/env python
# encoding: utf-8
# Created Time: 日 12/10 00:02:09 2017

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import keras
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from Hackthon.jddjr import utils


def build_model(input_seq_len, output_seq_len, num_samples, multi_gpus=False):

    RNN = layers.LSTM
    encoder_layers = 1
    decoder_layers = 2
    hidden_dim = 200
    model = models.Sequential()

    model.add(layers.TimeDistributed(layers.Dense(100, activation='relu'), 
        input_shape=(input_seq_len, 1)))
    for _ in range(encoder_layers):
        model.add(RNN(hidden_dim, return_sequences=True))
    model.add(RNN(hidden_dim, return_sequences=False))

    model.add(layers.RepeatVector(output_seq_len))
    for _ in range(decoder_layers):
        model.add(RNN(hidden_dim, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(1)))

    decay = 1. / num_samples
    optimizer = optimizers.Adam(lr=0.1, decay=decay)

    def score_func(y_true, y_pred):
        y_true = tf.reduce_sum(y_true, axis=1)
        y_pred = tf.reduce_sum(y_pred, axis=1)

        mae = tf.reduce_sum(tf.abs(y_true - y_pred))
        score = mae / tf.reduce_sum(y_true)
        return score

    if multi_gpus:
        model = keras.utils.multi_gpu_model(model, gpus=2)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    print('model input shape: {0}'.format(model.input_shape))
    print('model output shape: {0}'.format(model.output_shape))
    return model

def evaluate(model, datat):
    y_pred = model.predict(datat.valid_X)
    y_pred = np.squeeze(y_pred)
    y_pred = datat.scaler.inverse_transform(y_pred.T).T

    y_pred_sum = np.sum(y_pred, axis=1)
    y_true_sum = np.sum(datat.valid_Y, axis=1)
    score = np.sum(np.abs(y_pred_sum - y_true_sum)) / np.sum(y_true_sum)

    print('score: {0}'.format(score))
    return score



def main():
    input_seq_len = 90
    output_seq_len = 30
    batch_size = 64
    epochs = 100

    # step 1: 
    # utils.generate_csv_by_shop()

    # step 2: 
    # utils.generate_features_by_day(range(1, 3001))

    features = utils.get_features(range(1, 3001), 'features')
    print(features.shape)
    datat = utils.get_seq2seq_data_2(features, input_seq_len, output_seq_len, interval=2)
    num_samples = datat.train_X.shape[0]
    print('num_samples: {0}'.format(num_samples))

    model = build_model(input_seq_len, output_seq_len, num_samples, multi_gpus=False)
    log_dir = os.path.join(utils.DATA_DIR, 
            'logs_seq2seq/input_seq_len_{0}_output_seq_len_{1}'.format(
                input_seq_len, output_seq_len))

    callbacks = utils.get_callbacks(log_dir)

    print(model.summary())
    for ii in range(epochs):
        model.fit(datat.train_X, datat.train_Y, 
                batch_size=batch_size, epochs=ii+1, 
                callbacks=callbacks,
                initial_epoch=ii)
        evaluate(model, datat)




if __name__ == '__main__':
    main()
