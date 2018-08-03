# -*- coding: utf-8 -*-
"""
Created on Tue May 15 23:53:16 2018

@author: hangz
"""

import numpy as np
from keras.utils import np_utils
from readingUlits import readingUlits
from sklearn.cross_validation import train_test_split
from keras.applications.inception_v3 import InceptionV3,  decode_predictions
from keras.models import Model
from keras.layers import merge, Input, GlobalAveragePooling2D, Dense, Activation, Flatten
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import gc
import os.path
import csv
import time
import datetime
from keras.preprocessing.image import ImageDataGenerator


image_input = Input(shape=(299, 299, 3))

base_model = InceptionV3(input_tensor=image_input, weights='imagenet', include_top=False)



x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

predictions = Dense(20, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

model.layers.pop()

x = model.layers[-1].output

predictions = Dense(25, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

