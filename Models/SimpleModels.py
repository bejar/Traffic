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
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.models import Sequential
from keras.regularizers import l1, l2

__author__ = 'bejar'


def add_full_layer(model, fullsize, regfull, classes):
    """
    adds to the model a set of full layers and a final layer for the classes

    :param model:
    :param fullsize:
    :param regfull:
    :return:
    """
    for size in fullsize:
        if regfull[0] == 'l1':
            model.add(Dense(size, activation='relu', W_constraint=maxnorm(3), W_regularizer=l1(regfull[1])))
        if regfull[0] == 'l2':
            model.add(Dense(size, activation='relu', W_constraint=maxnorm(3), W_regularizer=l2(regfull[1])))
        if regfull[0] == 'drop':
            model.add(Dense(size, activation='relu', W_constraint=maxnorm(3)))
            model.add(Dropout(regfull[1]))

    model.add(Dense(classes, activation='softmax'))

def add_pooling(model, method, psize):
    """
    adds pooling to the model
    :param model:
    :param method:
    :param stride:
    :return:
    """
    if method == 'max':
        model.add(MaxPooling2D(pool_size=psize))
    if method == 'average':
        model.add(AveragePooling2D(pool_size=psize))

def simple_model(config):
    """
    Simple convolutional models
    :param smodel:
    :return:
    """
    input_shape = config['input_shape']
    num_classes = config['num_classes']

    convolayer = config['convolayers']['sizes']
    convofield = config['convolayers']['convofields']
    dropoutconvo = config['convolayers']['reg'][1]  # for now is always dropout
    pmethod = config['convolayers']['pool'][0]
    psize = (config['convolayers']['pool'][1], config['pool'][2])


    fulllayer = config['fulllayers']['sizes']
    regfull = config['fulllayers']['reg']

    smodel = config['model']

    if smodel == 1:
        # Model 1
        model = Sequential()
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], input_shape=input_shape, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], activation='relu', border_mode='same', W_constraint=maxnorm(3)))
        add_pooling(model, pmethod, psize)
        model.add(Flatten())
        add_full_layer(model, fulllayer, regfull, num_classes)
    elif smodel == 2:
        # Model 2 - Six convolutionals
        model = Sequential()
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], input_shape=input_shape, activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], activation='relu', border_mode='same'))
        add_pooling(model, pmethod, psize)
        model.add(Convolution2D(convolayer[-2], convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(convolayer[-2], convofield[1], convofield[1], activation='relu', border_mode='same'))
        add_pooling(model, pmethod, psize)
        model.add(Convolution2D(convolayer[-3], convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(convolayer[-3], convofield[1], convofield[1], activation='relu', border_mode='same'))
        add_pooling(model, pmethod, psize)
        model.add(Flatten())
        model.add(Dropout(dropoutconvo))
        add_full_layer(model, fulllayer, regfull, num_classes)
    elif smodel == 3:
        # Model 3 - Four convolutionals
        model = Sequential()
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], input_shape=input_shape, activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], activation='relu', border_mode='same'))
        add_pooling(model, pmethod, psize)
        model.add(Convolution2D(convolayer[-2], convofield[1], convofield[1], activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(convolayer[-2], convofield[1], convofield[1], activation='relu', border_mode='same'))
        add_pooling(model, pmethod, psize)
        model.add(Flatten())
        model.add(Dropout(dropoutconvo))
        add_full_layer(model, fulllayer, regfull, num_classes)
    elif smodel == 4:
        # Model 4 - Two convolutionals
        model = Sequential()
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], input_shape=input_shape, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(convolayer[-1], convofield[0], convofield[0], activation='relu', border_mode='same', W_constraint=maxnorm(3)))
        add_pooling(model, pmethod, psize)
        model.add(Flatten())
        model.add(Dropout(dropoutconvo))
        add_full_layer(model, fulllayer, regfull, num_classes)

    return model
