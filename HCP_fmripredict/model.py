#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

###define model for training
import sys
sys.path.append('/home/yuzhang/projects/rrg-pbellec/yuzhang/HCP/codes/HCP_fmripredict')
import utils

import numpy as np
import pandas as pd

from sklearn import svm, metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, DictionaryLearning, KernelPCA

try:
    from keras.utils import np_utils
    from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
    from keras.models import Model
except ImportError:
    print("Tensorflow is not avaliable in the current node!")
    print("deep learning models will not be running for this test!")


def build_fc_nn_model(Nfeatures,Nlabels,layers=3,hidden_size=256,dropout=0.25):
    ######fully-connected neural networks
    input0 = Input(shape=(Nfeatures,))
    drop1 = input0
    for li in np.arange(layers):
        hidden1 = Dense(hidden_size, activation='relu')(drop1)
        drop1 = Dropout(dropout)(hidden1)
        hidden_size = np.int32(hidden_size / 2)
        if hidden_size < 10:
            hidden_size = 16

    hidden2 = Dense(16, activation='relu')(drop1)
    drop2 = Dropout(0.5)(hidden2)
    out = Dense(Nlabels, activation='softmax')(drop2)

    model = Model(inputs=input0, outputs=out)
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse','accuracy'])
    model.summary()

    return model


def build_cnn_model(input_shape, Nlabels, filters=32, convsize=3, poolsize=2, hidden_size=128, conv_layers=2):
    #     import keras.backend as K
    #     if K.image_data_format() == 'channels_first':
    #         img_shape = (1,img_rows,img_cols)
    #     elif K.image_data_format() == 'channels_last':
    #         img_shape = (img_rows,img_cols,1)


    input0 = Input(shape=input_shape)
    drop1 = input0
    for li in range(conv_layers):
        conv1 = Conv2D(filters, (convsize, convsize), padding='same', activation='relu')(drop1)
        conv1 = Conv2D(filters, (convsize, convsize), padding='same', activation='relu')(conv1)
        pool1 = MaxPooling2D((poolsize, poolsize))(conv1)
        drop1 = Dropout(0.25)(pool1)
        filters *= 2


    drop2 = drop1
    flat = Flatten()(drop2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop3 = Dropout(0.5)(hidden)
    #hidden = Dense((hidden_size/4).astype(int), activation='relu')(drop3)
    #drop4 = Dropout(0.5)(hidden)
    out = Dense(Nlabels, activation='softmax')(drop3)

    model = Model(inputs=input0, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model
