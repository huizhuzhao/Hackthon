#!/usr/bin/env python
# encoding: utf-8
# Created Time: Thu 06 Jul 2017 09:40:01 PM CST


import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.preprocessing import image as keras_image


BATCH_SIZE = 128
TARGET_SIZE = (128, 128) # (height, width)
LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
LABELS_TO_INDEX = dict((l, i) for i, l in enumerate(LABELS))
INDEX_TO_LABELS = dict((i, l) for i, l in enumerate(LABELS))
CAPTION = {
    0: [u"不开心的山兔", u"有完没完", u'PP欠打', u'不嗨森', u'嘿我这暴脾气', u'怒怼'],
    1: [u'呕吐', u'恶心', u'吐出来你好恶心', u'讨厌的感觉', u'请自重'],
    2: [u'妈咪我怕', u'我的小心脏', u'怂了行不行', u'宝宝认怂', u'腿软暂时跑不掉'],
    3: [u'敲开心', u'美美的', u'哈哈哈', u'我的心花儿', u'开心到飞起'],
    4: [u'你说呢', u'我想静静', u'冷漠', u'不care', u'生无可恋', u'敢问路在何方'],
    5: [u'心碎成渣渣', u'桑心', u'宝宝不开心', u'求安慰', u'求抱抱'],
    6: [u'噢买噶', u'我去我去', u'吃惊', u'惊到了']}


def load_images(dir_path, grayscale=True, target_size=TARGET_SIZE):
    """
    load all the images in directory "dir_path"

    inputs
    ------
    dir_path: dir path where images are located
    grayscale: boolean, whether graycale or rgb arrays are returned
    target_size: tuple (height, width), the array size that every images are converted to
    """


    def _pointer_to_array(pointer, grayscale, target_size):
        img = keras_image.load_img(pointer, grayscale=grayscale, target_size=target_size)
        array = keras_image.img_to_array(img)
        return array

    names = [n for n in os.listdir(os.path.join(dir_path)) if n.endswith('jpg')]
    pointers = [os.path.join(dir_path, n) for n in names]

    arrays_list = []
    names_list = []
    failed_count = 0
    for p in pointers:
        try:
            array = _pointer_to_array(p, grayscale, target_size)
            arrays_list.append(array)
            names_list.append(os.path.basename(p).split('.')[0])
        except:
            failed_count += 1
            pass

    print('Failed: {0}, Total: {1}'.format(failed_count, len(pointers)))
    return np.asarray(arrays_list, dtype=arrays_list[0].dtype), names_list



class Generator(object):
    def __init__(self, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, color_mode='grayscale', shuffle=True):
        """
        inputs
        ------
        target_size: tuple (height, weight), do not has channel dim
        batch_size: integer
        color_mode: 'grayscale', 'rgb'
        shuffle: boolean
        """
        self.target_size = target_size
        self.batch_size = batch_size
        self.color_mode = color_mode
        self.shuffle = shuffle
        datagen = keras_image.ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)


        self.datagen = datagen

    def dir_generator(self, dir_path):
        generator = self.datagen.flow_from_directory(
                dir_path,
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                color_mode=self.color_mode,
                classes=LABELS,
                shuffle=self.shuffle)

        return generator

    def numpy_generator(self, X, Y=None):

        def rgb2gray(batch_array):
            assert len(batch_array.shape) == 4, (
                    'batch_array.shape: {0} received'.format(batch_array.shape))
            new_batch_array = []
            for array in batch_array:
                img = keras_image.array_to_img(array)
                if img.mode != 'L':
                    img = img.convert('L')
                    x = keras_image.img_to_array(img)
                    new_batch_array.append(x)
                    batch_array = np.asarray(new_batch_array, dtype=new_batch_array[0].dtype)

            return batch_array

        if self.color_mode == 'grayscale':
            X = rgb2gray(X)

        generator = self.datagen.flow(X, Y, shuffle=self.shuffle, batch_size=self.batch_size)
        return generator


def load_model(config_path, hdf5_path):
    config = json.load(open(config_path, 'r'))
    model = Sequential().from_config(config)
    model.load_weights(hdf5_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model


def get_callbacks(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, 'logger.csv')
    model_name = os.path.join(log_dir, 'VGG')
    model_name = model_name + '.{epoch:02d}--{val_acc:.2f}.hdf5'

    logger = CSVLogger(log_file, append=True)
    checkpoint = ModelCheckpoint(model_name, monitor='val_acc',
            verbose=1, save_best_only=True)
    callbacks = [checkpoint, logger]

    return callbacks


def dump_config(config, log_dir):
    path = os.path.join(log_dir, 'config.json')
    with open(path, 'w') as f_w:
        json.dump(config, f_w)




if __name__ == '__main__':
    dir_path = os.path.join(os.path.expanduser('~'), 'Hackthon/data-test')
    names = [n for n in os.listdir(os.path.join(dir_path)) if n.endswith('jpg')]
    pointers = [os.path.join(dir_path, n) for n in names]

    loader = Loader()
    array, names = loader.load_test_arrays(dir_path)
    #array = loader._pointer_to_array(pointers[0], False, (41, 41))
    print(array.shape)

    from Teemo.utils import cv2_utils

