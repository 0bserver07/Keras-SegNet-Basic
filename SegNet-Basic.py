from __future__ import absolute_import
from __future__ import print_function
import os

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'





import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization


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


# load the model:
with open('segNet_basic_model.json') as model_file:
    segnet_basic = models.model_from_json(model_file.read())



segnet_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

nb_epoch = 1
batch_size = 8

history = segnet_basic.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, class_weight=class_weighting )#, validation_data=(X_test, X_test))

# This save the trained model weights to this file with number of epochs
segnet_basic.save_weights('model_weight_{}.hdf5'.format(nb_epoch))

