"""
.. module:: SimpleModels

SimpleModels
******

:Description: SimpleModels

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  20/12/2016
"""

from keras.constraints import maxnorm
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential

__author__ = 'bejar'


def simple_model(smodel, config):
    """
    Simple convolotional models
    :param smodel:
    :return:
    """
    convofield = config['convofields']
    dropoutconvo = config['dpconvo']
    dropoutfull = config['dpfull']
    input_shape = config['input_shape']
    num_classes = config['num_classes']
    fulllayer = config['fulllayers']

    if smodel == 1:
        # Model 1
        model = Sequential()
        model.add(Convolution2D(32, convofield[0], convofield[0], input_shape=input_shape, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(32, convofield[0], convofield[0], activation='relu', border_mode='same', W_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
        model.add(Dense(fulllayer[-1], activation='relu', W_constraint=maxnorm(3))) #512
        model.add(Dropout(dropoutfull))
        model.add(Dense(num_classes, activation='softmax'))
    elif smodel == 2:
        # Model 2
        model = Sequential()
        model.add(Convolution2D(32, convofield[0], convofield[0], input_shape=input_shape, activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(32, convofield[0], convofield[0], activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(64, convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(128, convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(dropoutconvo))
        model.add(Dense(fulllayer[-2], activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutfull))
        model.add(Dense(fulllayer[-1], activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutfull))
        model.add(Dense(num_classes, activation='softmax'))
    elif smodel == 3:
        # Model 3
        model = Sequential()
        model.add(Convolution2D(32, convofield[0], convofield[0], input_shape=input_shape, activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(32, convofield[0], convofield[0], activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(64, convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(dropoutconvo))
        model.add(Dense(fulllayer[-2], activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutfull))
        model.add(Dense(fulllayer[-1], activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutfull))
        model.add(Dense(num_classes, activation='softmax'))
    elif smodel == 4:
        # Model 4
        model = Sequential()
        model.add(Convolution2D(32, convofield[0], convofield[0], input_shape=input_shape, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(32, convofield[0], convofield[0], activation='relu', border_mode='same', W_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
        model.add(Dropout(dropoutconvo))
        model.add(Dense(fulllayer[-2], activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutfull))
        model.add(Dense(fulllayer[-1], activation='relu', W_constraint=maxnorm(3))) #512
        model.add(Dropout(dropoutfull))
        model.add(Dense(num_classes, activation='softmax'))

    return model
