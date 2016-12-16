"""
.. module:: ConvoTest

ConvoTest
*************

:Description: ConvoTest

    

:Authors: bejar
    

:Version: 

:Created on: 28/11/2016 11:10 

"""

import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from Util.Generate_Dataset import generate_dataset
from Util.Logger import config_logger
import time
from Util.Cameras import Cameras
from sklearn.metrics import confusion_matrix, classification_report
#from keras.utils.visualize_util import plot
from Util.Constants import results_path
from Util.MyRemoteMonitor import MyRemoteMonitor
import socket

__author__ = 'bejar'


if __name__ == '__main__':



    seed = 7
    np.random.seed(seed)
    ltime = time.strftime('%Y%m%d%H%M%S', time.localtime(int(time.time())))
    log = config_logger(file='convolutional-' + ltime )
    ldaysTr = ['20161102','20161103','20161104','20161105','20161106','20161107','20161108','20161109','20161110','20161111', '20161112', '20161113', '20161114', '20161115', '20161117', '20161118', '20161119']
    ldaysTs = ['20161116']
    z_factor = 0.25
    camera = None  #'Ronda' #Cameras[0]

    log.info(' -- CNN ----------------------')
    log.info('Train= %s  Test= %s z_factor= %0.2f camera= %s', ldaysTr, ldaysTs, z_factor, camera)

    X_train, y_trainO, X_test, y_testO = generate_dataset(ldaysTr, ldaysTs, z_factor, PCA=False, method='two', reshape=False, cpatt=camera)
    X_train = X_train.transpose((0,3,1,2))
    X_test = X_test.transpose((0,3,1,2))

    y_trainO = [i -1 for i in y_trainO]
    y_testO = [i -1 for i in y_testO]
    y_train = np_utils.to_categorical(y_trainO, len(np.unique(y_trainO)))
    y_test = np_utils.to_categorical(y_testO, len(np.unique(y_testO)))
    num_classes = y_test.shape[1]
    print(num_classes)

    smodel = 3
    dropoutconvo = 0.2
    dropoutfull = 0.6
    convofield1 = 3
    convofield2 = 3
    if smodel == 1:
        # Model 1
        model = Sequential()
        model.add(Convolution2D(32, convofield1, convofield1, input_shape=X_train[0].shape, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(32, convofield1, convofield1, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu', W_constraint=maxnorm(3))) #512
        model.add(Dropout(dropoutfull))
        model.add(Dense(num_classes, activation='softmax'))
    elif smodel == 2:
        # Model 2
        model = Sequential()
        model.add(Convolution2D(32, convofield2, convofield2, input_shape=X_train[0].shape, activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(32, convofield2, convofield2, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, convofield1, convofield1, activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(64, convofield1, convofield1, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, convofield1, convofield1, activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(128, convofield1, convofield1, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(dropoutconvo))
        model.add(Dense(64, activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutfull))
        model.add(Dense(32, activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutfull))
        model.add(Dense(num_classes, activation='softmax'))
    elif smodel == 3:
        # Model 3
        model = Sequential()
        model.add(Convolution2D(32, convofield2, convofield2, input_shape=X_train[0].shape, activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(32, convofield2, convofield2, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, convofield1, convofield1, activation='relu', border_mode='same'))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(64, convofield1, convofield1, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(dropoutconvo))
        model.add(Dense(128, activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutfull))
        model.add(Dense(64, activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutfull))
        model.add(Dense(num_classes, activation='softmax'))
    elif smodel == 4:
        # Model 4
        model = Sequential()
        model.add(Convolution2D(32, convofield1, convofield1, input_shape=X_train[0].shape, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutconvo))
        model.add(Convolution2D(32, convofield1, convofield1, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
        model.add(Dropout(dropoutconvo))
        model.add(Dense(64, activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(dropoutfull))
        model.add(Dense(32, activation='relu', W_constraint=maxnorm(3))) #512
        model.add(Dropout(dropoutfull))
        model.add(Dense(num_classes, activation='softmax'))

    # Compile model

    epochs = 100
    lrate = 0.005 #0.01
    momentum = 0.9
    batchsize = 100
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=momentum, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    log.info('Model = %d, Epochs= %d, LRate= %.3f, Momentum= %.2f', smodel, epochs, lrate, momentum)
    log.info('DPC = %.2f, DPF = %.2f, Batch Size= %d', dropoutconvo, dropoutfull, batchsize)
    model.summary()
    log.info('%s', model.to_json())
    log.info('BEGIN= %s',time.strftime('%d-%m-%Y %H:%M:%S', time.localtime()))

    remote = MyRemoteMonitor(id='%sMd%d-Ep%d-LR%.3f-MM%.2f-DPC%.2f-DPF%.2f-BS%d'%
                                (socket.gethostname(), smodel, epochs, lrate, momentum,dropoutconvo, dropoutfull, batchsize),
                            root='http://chandra.cs.upc.edu',
                            path='/Update')

    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=batchsize, callbacks=[remote],
                     class_weight={0: 1.0, 1: 1.5, 2: 2.0, 3: 3.0, 4: 4.0}) 
    log.info('END= %s',time.strftime('%d-%m-%Y %H:%M:%S', time.localtime()))

    log.info('%s', hist.history)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    log.info("Accuracy: %.2f%%",(scores[1]*100))
    labels = model.predict(X_test)

    y_pred = model.predict_classes(X_test)

    p = model.predict_proba(X_test) # to predict probability
    log.info('%s',confusion_matrix(y_testO, y_pred))

    log.info('%s',classification_report(y_testO, y_pred))

