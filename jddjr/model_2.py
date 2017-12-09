#!/usr/bin/env python
# encoding: utf-8
# Created Time: 日 12/10 00:02:09 2017


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from Hackthon.jddjr import utils


def build_model(seq_len):

    RNN = layers.LSTM
    LAYERS = 3
    model = models.Sequential()
    for _ in range(LAYERS):
        model.add(RNN(100, input_shape=(seq_len, 1), return_sequences=True))

    model.add(RNN(100, return_sequences=False))
    model.add(layers.RepeatVector(90))
    model.add(layers.TimeDistributed(layers.Dense(1)))

    optimizer = optimizers.Adam(lr=0.01)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    print('model input shape: {0}'.format(model.input_shape))
    print('model output shape: {0}'.format(model.output_shape))
    return model


def main():
    seq_len = 181
    batch_size = 32
    epochs = 20
    sale_amt = utils.get_sale_amt_by_day(range(1, 300))
    datat = utils.DataTransform(sale_amt, seq_len)
    datat.scale()
    datat.get_train_data()

    model = build_model(seq_len)


if __name__ == '__main__':
    main()
