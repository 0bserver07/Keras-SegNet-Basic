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



val_data = np.load('./data/val_data.npy')
val_label = np.load('./data/val_label.npy')

# load the model:
with open('segNet_basic_model.json') as model_file:
    segnet_basic = models.model_from_json(model_file.read())


# load weights
segnet_basic.load_weights("weights.best.hdf5")

# Compile model (required to make predictions)
segnet_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

batch_size = 6

# estimate accuracy on whole dataset using loaded weights
scores = segnet_basic.evaluate(val_data, val_label, verbose=0, batch_size=batch_size)
print("%s: %.2f%%" % (segnet_basic.metrics_names[1], scores[1]*100))
