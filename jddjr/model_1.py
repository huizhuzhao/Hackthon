#!/usr/bin/env python
# encoding: utf-8
# Created Time: 六 12/ 9 21:57:07 2017

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import layers
from Hackthon.jddjr import utils


def build_model(seq_len):

    RNN = keras.layers.LSTM
    LAYERS = 3
    model = keras.models.Sequential()
    for _ in range(LAYERS):
        model.add(RNN(100, input_shape=(seq_len, 1), return_sequences=True))

    model.add(RNN(100, return_sequences=False))
    model.add(layers.Dense(1, activation='linear'))

    optimizer = optimizers.Adam(lr=0.01)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    print('model input shape: {0}'.format(model.input_shape))
    print('model output shape: {0}'.format(model.output_shape))
    return model


def evaluate(model, valid_X, valid_Y, seq_len, scaler=None):
    input_X = valid_X
    pred_Y = []
    for ii in range(90):
        pred_y = model.predict(input_X)
        pred_Y.append(pred_y)
        pred_y = pred_y[:, :, np.newaxis]
        input_X = np.concatenate([input_X[:, 1:], pred_y], axis=1)

    pred_Y = np.concatenate(pred_Y, axis=1)
    if scaler is not None:
        pred_Y = scaler.inverse_transform(pred_Y.T).T

    print('pred_Y.shape: {0}'.format(pred_Y.shape))
    print('valid_Y.shape: {0}'.format(valid_Y.shape))
    assert pred_Y.shape == valid_Y.shape
    pred_Y = np.sum(pred_Y, axis=1)
    valid_Y = np.sum(valid_Y, axis=1)
    score = np.sum(np.abs(pred_Y - valid_Y)) / np.sum(valid_Y)
    print('score: {0}'.format(score))
    return score



def main():
    seq_len = 10
    batch_size = 32
    epochs = 20
    sale_amt = utils.get_sale_amt_by_day(range(1, 3001))
    datat = utils.DataTransform(sale_amt, seq_len)
    datat.scale()
    datat.get_train_data()

    model = build_model(seq_len)
    scores = []
    for ii in range(epochs):
        score = evaluate(model, datat.valid_X, datat.valid_Y, seq_len, datat.scaler)
        scores.append(score)
        model.fit(datat.train_X, datat.train_Y, batch_size=batch_size, epochs=1)

    df_metrics = pd.DataFrame({'epochs': range(epochs), 'scores': scores})
    df_metrics.to_csv(os.path.join(utils.DATA_DIR, 'logger.csv'))




if __name__ == '__main__':
    main()
