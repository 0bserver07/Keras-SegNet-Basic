from __future__ import absolute_import
from __future__ import print_function
import os

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'





import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

from keras import backend as K

import cv2
import numpy as np
import json
np.random.seed(07) # 0bserver07 for reproducibility


data_shape = 360*480

class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]


# load the data
train_data = np.load('./data/train_data.npy')
train_label = np.load('./data/train_label.npy')

test_data = np.load('./data/test_data.npy')
test_label = np.load('./data/test_label.npy')

# load the model:
with open('segNet_basic_model.json') as model_file:
    segnet_basic = models.model_from_json(model_file.read())


segnet_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

nb_epoch = 100
batch_size = 6

# Fit the model
history = segnet_basic.fit(train_data, train_label, callbacks=callbacks_list, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, class_weight=class_weighting , validation_data=(test_data, test_label), shuffle=True) # validation_split=0.33

# This save the trained model weights to this file with number of epochs
segnet_basic.save_weights('weights/model_weight_{}.hdf5'.format(nb_epoch))

