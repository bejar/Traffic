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


def add_full_layer(model, fullsize, dropout, classes):
    """
    adds to the model a set of full layers and a final layer for the classes

    :param model:
    :param fullsize:
    :param dropout:
    :return:
    """
    for size in fullsize:
        model.add(Dense(size, activation='relu', W_constraint=maxnorm(3)))
        if dropout:
            model.add(Dropout(dropout))
    model.add(Dense(classes, activation='softmax'))


def simple_model(config):
    """
    Simple convolutional models
    :param smodel:
    :return:
    """
    convofield = config['convofields']
    dropoutconvo = config['dpconvo']
    dropoutfull = config['dpfull']
    input_shape = config['input_shape']
    num_classes = config['num_classes']
    fulllayer = config['fulllayers']
    convolayer = config['convolayers']
    smodel = config['model']

    if smodel == 1:
        # Model 1
        model = Sequential()
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], input_shape=input_shape, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], activation='relu', border_mode='same', W_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
        add_full_layer(model, fulllayer, dropoutfull, num_classes)
    elif smodel == 2:
        # Model 2
        model = Sequential()
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], input_shape=input_shape, activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(convolayer[-2], convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(convolayer[-2], convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(convolayer[-3], convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(convolayer[-3], convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(dropoutconvo))
        add_full_layer(model, fulllayer, dropoutfull, num_classes)
    elif smodel == 3:
        # Model 3
        model = Sequential()
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], input_shape=input_shape, activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(convolayer[-2], convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(convolayer[-2], convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(dropoutconvo))
        add_full_layer(model, fulllayer, dropoutfull, num_classes)
    elif smodel == 4:
        # Model 4
        model = Sequential()
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], input_shape=input_shape, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], activation='relu', border_mode='same', W_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(dropoutconvo))
        add_full_layer(model, fulllayer, dropoutfull, num_classes)


    return model
