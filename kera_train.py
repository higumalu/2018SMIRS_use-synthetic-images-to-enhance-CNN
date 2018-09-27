# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:51:55 2018

@author: HIGUMA_LU
"""

import sys
import os
from sklearn.metrics import roc_curve, auc
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import callbacks

DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True

if DEV:
  epochs = 4
else:
  epochs = 72


train_data_dir = './data/train_mix'
validation_data_dir = './data/validation'

img_width, img_height = 180, 120
#nor:800 abnor: 300,1500,1800 total:1100,2300,2600
nb_train_samples = 1100 
nb_validation_samples = 400
nb_filters1 = 32
nb_filters2 = 64
nb_filters3 = 128
conv1_size = 5
conv2_size = 3
pool_size = 3
classes_num = 2
batch_size = 64
lr = 0.0004

model = Sequential()
model.add(Conv2D(nb_filters1, (conv1_size, conv1_size), padding="same", input_shape=(img_width, img_height, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.5))

model.add(Conv2D(nb_filters2, (conv2_size, conv2_size), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format='channels_first'))
model.add(Dropout(0.5))

model.add(Conv2D(nb_filters3, (conv2_size, conv2_size), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format='channels_first'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1. / 255)

test_datagen = ImageDataGenerator(
    rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical')

"""
Tensorboard log
tensorboard --logdir=
"""
log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, 
                              histogram_freq=0,
                              batch_size=64,
                              write_graph=True)
cbks = [tb_cb]


model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator ,
    validation_steps=nb_validation_samples,
    callbacks=cbks)



target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models/model.h5')
model.save_weights('./models/weights.h5')


