# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:57:39 2020

@author: Anirudh
"""


# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
import keras

class TrafficSignNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        model.add(Conv2D(32, (3, 3),input_shape=inputShape, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        
        model.add(Conv2D(32, (3, 3), padding="same", activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
      
        
        
     
        model.add(Flatten())
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dropout(rate = 0.1))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        #,kernel_regularizer=keras.regularizers.l1_l2(l1=0.1,l2=0.01
        
        model.add(Dense(classes, activation = 'softmax'))
        
        return model
