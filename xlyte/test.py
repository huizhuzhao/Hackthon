#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年07月16日 星期日 10时37分57秒

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import keras_model


TRAIN_DIR = './data/data-train'
TEST_DIR = './data/data-test'
TRUE_TEST_LABELS = './data/expression_test_result.csv'
config_path = './logs/config.json'
hdf5_path = './logs/VGG.390--0.93.hdf5'


def train_data():
    """
    >>> train_gen = train_data()
    >>> batch_x, batch_y = train_gen.next()
    >>> print(batch_x.shape, batch_y.shape)
    (64, 128, 128, 1), (64, 7)
    """
    generator = utils.Generator(target_size=utils.TARGET_SIZE,
                                color_mode='grayscale',
                                batch_size=utils.BATCH_SIZE)

    train_gen = generator.dir_generator(TRAIN_DIR)
    return train_gen

def test_data():
    """
    >>> X_test, y_test = test_data()
    >>> print(X_test.shape, y_test.shape)
    (3036, 128, 128, 1), (3036, 7)
    """
    ## prepare test data
    arrays, image_ids = utils.load_images(TEST_DIR,
                                          grayscale=True,
                                          target_size=utils.TARGET_SIZE)
    generator = utils.Generator(target_size=utils.TARGET_SIZE,
                                color_mode='grayscale',
                                batch_size=len(image_ids),
                                shuffle=False)


    test_gen = generator.numpy_generator(arrays)
    X_test = test_gen.next()

    # the ground true label
    df = pd.read_csv(TRUE_TEST_LABELS)
    id_to_label = dict(zip(df.image_id.tolist(), df.label.tolist()))
    y_test_true = np.asarray([id_to_label[id] for id in image_ids], dtype=np.int32)
    y_test_true = np.eye(7)[y_test_true]

    return X_test, y_test_true


def test_model():
    ## load trained model
    model = utils.load_model(config_path, hdf5_path)
    y_pred = model.predict(X, verbose=True)
    y_pred = np.argmax(y_pred, axis=1).astype(np.float32)

    # compute accuracy
    accu = np.sum(y_pred == y_true) * 1. / len(y_true)
    print('accu: {0}'.format(accu))


    #val_res = model.evaluate(batch_x, batch_y)
    #print(val_res)

def train_model():
    train_gen = train_data()
    X_test, y_test = test_data()

    model = keras_model.vgg(input_shape=utils.TARGET_SIZE+(1,))
    print(model.input_shape)
    print(model.output_shape)

    callbacks = utils.get_callbacks('./logs')
    model.fit_generator(train_gen,
                        steps_per_epoch=6000/utils.BATCH_SIZE,
                        epochs=500,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(X_test, y_test),
                        validation_steps=len(X_test)/utils.BATCH_SIZE)


def main():
    logger_path = './logs/logger.csv'
    df = pd.read_csv(logger_path)
    plt.plot(df.epoch, df.acc, label='training', c='r')
    plt.plot(df.epoch, df.val_acc, label='validation', c='b')
    plt.legend(loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()



if __name__ == '__main__':
    main()
