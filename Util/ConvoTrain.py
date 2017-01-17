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


from keras import backend as K

from keras.optimizers import SGD
from keras.utils import np_utils
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


def train_model_batch(model, config, test, test_labels, acctrain=False):
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
            X_train, y_train, perm = dayGenerator(config['datapath'], day, config['zfactor'], config['num_classes'], config['batchsize'], reb=reb, imgord=config['imgord'])
            for p in perm:
                loss = model.train_on_batch(X_train[p], y_train[p], class_weight=classweight)
                tloss.append(loss[0])
                tacc.append(loss[1])

        # If acctrain is true then test all the train with the retrained model to obtain the real loss and acc after training
        # in the end the real measure of generalization is obtained with the independent test
        if acctrain:
            tloss = []
            tacc = []
            # Test Batches
            for day in ldaysTr:
                X_train, y_train, perm = dayGenerator(config['datapath'], day, config['zfactor'], config['num_classes'], config['batchsize'], reb=reb, imgord=config['imgord'])
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


def load_dataset(config, only_test=False, imgord='th'):
    """
    Loads the train and test dataset

    :return:
    """
    ldaysTr = config['train']
    ldaysTs = config['test']
    z_factor = config['zfactor']
    datapath = config['datapath']

    if not only_test:
        X_train, y_trainO = load_generated_dataset(datapath, ldaysTr, z_factor)
        if imgord == 'th':
            X_train = X_train.transpose((0,3,1,2))
        y_trainO = [i - 1 for i in y_trainO]
        y_train = np_utils.to_categorical(y_trainO, len(np.unique(y_trainO)))
    else:
        X_train = None,
        y_train = None

    X_test, y_testO = load_generated_dataset(datapath, ldaysTs, z_factor)

    if imgord == 'th':
        X_test = X_test.transpose((0,3,1,2))
    y_testO = [i -1 for i in y_testO]
    y_test = np_utils.to_categorical(y_testO, len(np.unique(y_testO)))


    num_classes = y_test.shape[1]

    return (X_train, y_train), (X_test, y_test), y_testO, num_classes


def load_generated_dataset(datapath, ldaysTr, z_factor):
    """
    Load the already generated datasets

    :param ldaysTr:
    :param ldaysTs:
    :param z_factor:
    :return:
    """
    ldata = []
    y_train = []
    for day in ldaysTr:
        data = np.load(datapath + 'data-D%s-Z%0.2f.npy' % (day, z_factor))
        ldata.append(data)
        y_train.extend(np.load(datapath + 'labels-D%s-Z%0.2f.npy' % (day, z_factor)))
    X_train = np.concatenate(ldata)

    return X_train, y_train

