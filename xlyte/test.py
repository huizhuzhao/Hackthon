#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年07月16日 星期日 10时37分57秒

import os
import numpy as np
import pandas as pd

from Hackthon.xlyte import utils
from Utils import pil_utils


USER_PATH = os.path.expanduser('~')
XLYTE_DIR = os.path.join(USER_PATH, 'bitbucket/test_data/xlyte')
TRAIN_DIR = os.path.join(USER_PATH, 'bitbucket/test_data/xlyte/data-all')
TEST_DIR = os.path.join(USER_PATH, 'bitbucket/test_data/xlyte/data-test')
TEST_LABELS = os.path.join(USER_PATH, 'bitbucket/test_data/xlyte/expression_test_result.csv')
TEST_LABELS2 = os.path.join(USER_PATH, 'Hackthon/test_results_3036.csv')
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def train_dataset():
    generator = utils.Generator(target_size=(128, 128), color_mode='grayscale', batch_size=10000)
    dir_gen = generator.dir_generator(TRAIN_DIR)
    batch_x, batch_y = dir_gen.next()

    config_path = '/home/xtalpi/Hackthon/logs/VGG_02/config.joblib'
    hdf5_path = '/home/xtalpi/Hackthon/logs/VGG_02/VGG.390--0.93.hdf5'
    model = utils.load_model(config_path, hdf5_path)

    #val_res = model.evaluate(batch_x, batch_y)
    #print(val_res)

def test_dataset():

    arrays, names = utils.load_images(TEST_DIR, grayscale=True, target_size=(128, 128))
    generator = utils.Generator(target_size=(128, 128), color_mode='grayscale', batch_size=10000, shuffle=False)
    test_gen = generator.numpy_generator(arrays)

    batch_x = test_gen.next()

    config_path = '/home/xtalpi/Hackthon/logs/VGG_02/config.joblib'
    hdf5_path = '/home/xtalpi/Hackthon/logs/VGG_02/VGG.390--0.93.hdf5'
    model = utils.load_model(config_path, hdf5_path)
    y_pred = model.predict(batch_x, verbose=True)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred.shape)

    df = pd.read_csv(TEST_LABELS)
    id_to_label = dict(zip(df.image_id.tolist(), df.label.tolist()))

    true_n = 0
    for k, v in zip(names, y_pred):
        if id_to_label[k] == v:
            true_n += 1

    print(true_n, len(y_pred))
    print(true_n * 1./len(names))





if __name__ == '__main__':
    test_dataset()


