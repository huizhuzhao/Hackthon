#!/usr/bin/env python
# encoding: utf-8
# author: huizhu
# created time: 2017年07月16日 星期日 10时37分57秒

import os
import numpy as np


USER_PATH = os.path.expanduser('~')
XLYTE_DIR = os.path.join(USER_PATH, 'bitbucket/test_data/xlyte')
TRAIN_DIR = os.path.join(USER_PATH, 'bitbucket/test_data/xlyte/data-all')
TEST_DIR = os.path.join(USER_PATH, 'bitbucket/test_data/xlyte/data-test')
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def foo():
    num = 0
    for e in EMOTIONS:
        sub_dir = os.path.join(TRAIN_DIR, e)
        names = [x for x in os.listdir(sub_dir) if x.endswith('jpg')]
        print(e, len(names))
        num += len(names)

    print(num)

    test_names = [x for x in os.listdir(TEST_DIR) if x.endswith('jpg')]
    print('test')
    print(len(test_names))


if __name__ == '__main__':
    foo()


