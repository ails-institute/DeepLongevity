# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:20:38 2020

@author: Artur Yakimovich, PhD
"""

from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, SpatialDropout2D
from keras.regularizers import l1, l2
from keras.constraints import max_norm
def getWormNet(img_shape=(900,900,1), n_classes=2, pooling='max', img_format='channels_last', dilate_by=1):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=11, strides=6, dilation_rate=dilate_by, activation='relu', padding='same', input_shape=img_shape, data_format=img_format))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())
    #model.add(Conv2D(64, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D())
   # model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(SpatialDropout2D(0.2))    
    model.add(Conv2D(256, kernel_size=5, activation='relu'))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())
   # model.add(BatchNormalization())
    model.add(MaxPooling2D())
   # model.add(MaxPooling2D(pool_size=2, strides=2))
    
    model.add(Conv2D(384, kernel_size=3, activation='relu'))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())
    
    model.add(Conv2D(384, kernel_size=3, activation='relu'))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())
   # model.add(BatchNormalization())
   
    model.add(MaxPooling2D())
   # model.add(MaxPooling2D(pool_size=2, strides=2))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_constraint=max_norm(1.0)))
    model.add(BatchNormalization())
    model.add(Dense(n_classes, activation='softmax'))
    
    return model