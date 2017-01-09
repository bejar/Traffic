"""
.. module:: ConvoTrain

ConvoTrain
*************

:Description: ConvoTrain

    

:Authors: bejar
    

:Version: 

:Created on: 20/12/2016 14:16 

"""

__author__ = 'bejar'


import numpy as np
from keras import backend as K

from keras.optimizers import SGD
from keras.utils import np_utils
from Util.Generate_Dataset import generate_dataset, load_generated_dataset
from sklearn.metrics import confusion_matrix, classification_report
from Util.DBLog import DBLog
from Util.DBConfig import mongoconnection
from Util.DataGenerators import dayGenerator
from numpy.random import shuffle
import numpy as np
__author__ = 'bejar'

K.set_image_dim_ordering('th')

def transweights(weights):
    wtrans = {}
    for v in weights:
        wtrans[str(v)] = weights[v]
    return wtrans

def detransweights(weights):
    wtrans = {}
    for v in weights:
        wtrans[int(v)] = weights[v]
    return wtrans

def train_model(model, config, train, test, test_labels, generator=None, samples_epoch=10000):
    """
    Trains the model

    :return:
    """
    sgd = SGD(lr=config['lrate'], momentum=config['momentum'], decay=config['lrate']/config['momentum'], nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    classweight = detransweights(config['classweight'])
    dblog = DBLog(database=mongoconnection, config=config, model=model, modelj=model.to_json())

    if generator is None:
        model.fit(train[0], train[1], validation_data=(test[0], test[1]), nb_epoch=config['epochs'],
                  batch_size=config['batchsize'], callbacks=[dblog], class_weight=classweight, verbose=0)
    else:
        model.fit_generator(generator, samples_per_epoch=samples_epoch, validation_data=(test[0], test[1]), nb_epoch=config['epochs'],
                   callbacks=[dblog], class_weight=classweight, verbose=0)


    scores = model.evaluate(test[0], test[1], verbose=0)
    y_pred = model.predict_classes(test[0], verbose=0)

    dblog.save_final_results(scores, confusion_matrix(test_labels, y_pred), classification_report(test_labels, y_pred))


def train_model_batch(model, config, test, test_labels):
    """
    Trains the model using Keras batch method

    :param model:
    :param config:
    :param test:
    :param test_labels:
    :return:
    """

    sgd = SGD(lr=config['lrate'], momentum=config['momentum'], decay=config['lrate']/config['momentum'], nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    classweight = detransweights(config['classweight'])
    dblog = DBLog(database=mongoconnection, config=config, model=model, modelj=model.to_json())

    ldaysTr = config['train']
    reb = config['rebalanced']
    # Train Epochs
    logs = {'loss':0.0, 'acc':0.0, 'val_loss':0.0, 'val_acc':0.0}
    for epoch in range(config['epochs']):
        shuffle(ldaysTr)
        tloss = []
        tacc = []
        # Train Batches
        for day in ldaysTr:
            X_train, y_train, perm, _ = dayGenerator(day, config['zfactor'], config['num_classes'], config['batchsize'], reb=reb)
            for p in perm:
                loss = model.train_on_batch(X_train[p], y_train[p], class_weight=classweight)
                tloss.append(loss[0])
                tacc.append(loss[1])
        #print('Loss %2.3f Acc %2.3f' % (np.mean(tloss), np.mean(tacc)))
        # logs['loss'] = float(np.mean(tloss))
        # logs['acc'] = float(np.mean(tacc))

        # Test Batches
        for day in ldaysTr:
            X_train, _, perm, y_train = dayGenerator(day, config['zfactor'], config['num_classes'], config['batchsize'], reb=reb)
            for p in perm:
                loss = model.test_on_batch(X_train[p], y_train[p])
                tloss.append(loss[0])
                tacc.append(loss[1])

        logs['loss'] = float(np.mean(tloss))
        logs['acc'] = float(np.mean(tacc))

        scores = model.evaluate(test[0], test[1], verbose=0)
        logs['val_loss'] = scores[0]
        logs['val_acc'] = scores[1]

        dblog.on_epoch_end(epoch, logs=logs)

    scores = model.evaluate(test[0], test[1], verbose=0)
    dblog.on_train_end(logs={'acc':logs['acc'], 'val_acc':scores[1]})
    y_pred = model.predict_classes(test[0], verbose=0)
    dblog.save_final_results(scores, confusion_matrix(test_labels, y_pred), classification_report(test_labels, y_pred))


def load_dataset(ldaysTr, ldaysTs, z_factor, gen=True, only_test=False):
    """
    Loads the train and test dataset

    :return:
    """

    if not only_test:
        if gen:
            X_train, y_trainO = generate_dataset(ldaysTr,z_factor, method='two')
        else:
            X_train, y_trainO = load_generated_dataset(ldaysTr, z_factor)
        X_train = X_train.transpose((0,3,1,2))
        y_trainO = [i - 1 for i in y_trainO]
        y_train = np_utils.to_categorical(y_trainO, len(np.unique(y_trainO)))
    else:
        X_train = None,
        y_train = None

    if gen:
        X_test, y_testO = generate_dataset(ldaysTs, z_factor, method='two')
    else:
        X_test, y_testO = load_generated_dataset(ldaysTs, z_factor)


    X_test = X_test.transpose((0,3,1,2))
    y_testO = [i -1 for i in y_testO]
    y_test = np_utils.to_categorical(y_testO, len(np.unique(y_testO)))


    num_classes = y_test.shape[1]

    return (X_train, y_train), (X_test, y_test), y_testO, num_classes

