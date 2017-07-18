#!/usr/bin/env python
# encoding: utf-8
# Created Time: Thu 06 Jul 2017 09:40:01 PM CST
# by wangmiao & huizhu


import os
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import cv2
from PIL import Image, ImageFont, ImageDraw
import imageio


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

COLOR = [
    (255, 255, 255),
    (255, 0, 0),
    (255, 122, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255)]



################# train model utils ###############
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


################# image process ###############
def extract_frames(ingif, outfolder):
    '''
    ingif, gif file
    outfolder: directory path to store extracted images
    '''
    frame = Image.open(ingif)
    nframes = 0
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    while frame:
        background = Image.new("RGB", frame.size, (255, 255, 255))
        background.paste(frame)
        background.save('%s/%s-%s.jpg' % (outfolder, os.path.basename(ingif), nframes), 'JPEG')
        nframes += 1
        try:
            frame.seek(nframes)
        except EOFError:
            break
    return True


def add_watermark(in_pic_name, watermark_str, x_idx, y_idx, out_dir):
    '''
    in_pic_name: string object, input picture name
    out_pic_name: string object, output picture name
    watermark_str: string
    point to draw
    y_idx: row
    x_idx: col
     0/0---X--->
    |
    |
    Y
    |
    |
    v

    '''
    with Image.open(in_pic_name).convert('RGBA') as im:
        row, col = im.size
        watermark = Image.new(im.mode, im.size)
        d_s = ImageDraw.Draw(watermark)
        font1 = ImageFont.truetype('./gif/Noto.otf', size=30)
        color = random.choice(COLOR)
        d_s.text((y_idx, x_idx), watermark_str, fill=color, font=font1)
        out = Image.alpha_composite(im, watermark)
        jpg_name = os.path.basename(in_pic_name)
        out_jpg_path = os.path.join(out_dir, jpg_name)
        out.save(out_jpg_path)

    return None


def cut_out_face(jpg_name):
    '''
    get jpg_name and cut it out
    only select one face
    '''
    face_cascade = cv2.CascadeClassifier('./gif/haarcascade_frontalface_default.xml')
    img = cv2.imread(jpg_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    image_faces = []
    locations = []
    for (x, y, w, h) in faces:
         #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
         image_faces.append(img[y: y+h, x:x+w])
         locations.append((x,y,w,h))

    if image_faces:
        # resize
        new_face = cv2.resize(image_faces[0], TARGET_SIZE[:2])
        # save
        #cv2.imwrite(jpg_name.strip('.jpg')+'face.jpg', new_face)
        return new_face, (x,y,w,h)
    else:
        return None


def detection(face_array, model):
    '''
    inputs
    ------
    face_array: shape=(height, width, channel)
    return label of face
    '''
    X_test = face_array[np.newaxis, :, :, :]

    generator = Generator(target_size=TARGET_SIZE,
                                color_mode='grayscale',
                                batch_size=BATCH_SIZE)

    test_gen = generator.numpy_generator(X_test)
    X_test = test_gen.next()
    y_pred = model.predict(X_test)
    label = np.argmax(y_pred)

    return label


def run(gif_file, model):
    '''
    pass
    '''
    # extract jpg files from gif in directory 'extract'
    base_path = os.path.abspath('.')
    extract_path = os.path.join(base_path, 'gif/extract')
    extract_frames(gif_file, extract_path)

    # frame_jpgs
    frame_jpgs = sorted(
        [f for f in os.listdir(extract_path) if f.endswith('.jpg')])
    num_frames = len(frame_jpgs)

    caption_location = []
    for idx, jpg_name in enumerate(frame_jpgs):
        feed_jpg = os.path.join(extract_path, jpg_name)
        result = cut_out_face(feed_jpg)
        if result is None:
            caption_location.append(None)
        else:
            face_array, location = result
            expression_num = detection(face_array, model)
            caption = random.choice(CAPTION[expression_num])
            caption_location.append((caption, location))

    # get caption and location
    # generate caption pictures
    print('caption_location')
    print(caption_location)
    print('len(caption_location', len(caption_location))
    out_path = os.path.join(base_path, 'gif/out')

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for capt_loc, jpg_name in zip(caption_location, frame_jpgs):
        jpg_path = os.path.join(extract_path, jpg_name)
        if capt_loc is None:
            dst = os.path.join(out_path, jpg_name)
            shutil.copyfile(jpg_path, dst)
        else:
            cap_str, loc = capt_loc
            x,y,w,h = loc
            add_watermark(jpg_path, cap_str, y+h, x, out_path)

    # generate GIFs
    file_names = [os.path.join(out_path, "{}-{}.jpg".format(os.path.basename(gif_file), i)) 
                    for i in range(num_frames)]
    print(file_names)
    gif_path = os.path.join(base_path, 'gif/out/out.gif')
    with imageio.get_writer(gif_path, mode='I') as writer:
        for filename in file_names:
            image = imageio.imread(filename)
            writer.append_data(image)

    return gif_path



if __name__ == '__main__':
    out_dir = './gif/images'

    #extract_frames(gif_path, out_dir)
    #add_watermark(img_path, string, 0, 0, out_dir)
    #face, idx = cut_out_face(img_path)
    config_path = './logs/config.json'
    hdf5_path = './logs/VGG.390--0.93.hdf5'
    model = load_model(config_path, hdf5_path)
    #detection(face, model)
    #cv2_utils.show_array(face)
    gif_path = run('./gif/g9.gif', model)
    print(gif_path)
