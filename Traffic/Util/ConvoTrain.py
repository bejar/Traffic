"""
.. module:: ConvoTrain

ConvoTrain
*************

:Description: ConvoTrain

    

:Authors: bejar
    

:Version: 

:Created on: 20/12/2016 14:16 

"""


import numpy as np
from Traffic.Callback.DBLog import DBLog
from Traffic.Util.DataGenerators import dayGenerator
from Traffic.Private.DBConfig import mongoconnection
from keras import backend as K
from keras.optimizers import SGD, Adagrad, Adadelta, Adam
from keras.utils import np_utils
from numpy.random import shuffle
from sklearn.metrics import confusion_matrix, classification_report

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

    if config['optimizer']['method'] == 'adagrad':
        optimizer = Adagrad()
    elif config['optimizer']['method'] == 'adadelta':
        optimizer = Adadelta()
    elif config['optimizer']['method'] == 'adam':
        optimizer = Adam()
    else:  # default SGD
        params = config['optimizer']['method']['params']
        optimizer = SGD(lr=params['lrate'], momentum=params['momentum'], decay=params['decay'],
                        nesterov=params['nesterov'])

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    classweight = detransweights(config['train']['classweight'])
    dblog = DBLog(database=mongoconnection, config=config, model=model, modelj=model.to_json())

    if generator is None:
        model.fit(train[0], train[1], validation_data=(test[0], test[1]), nb_epoch=config['train']['epochs'],
                  batch_size=config['train']['batchsize'], callbacks=[dblog], class_weight=classweight, verbose=0)
    else:
        model.fit_generator(generator, samples_per_epoch=samples_epoch, validation_data=(test[0], test[1]),
                            nb_epoch=config['train']['epochs'],
                            callbacks=[dblog], class_weight=classweight, verbose=0)

    scores = model.evaluate(test[0], test[1], verbose=0)
    y_pred = model.predict_classes(test[0], verbose=0)

    dblog.save_final_results(scores, confusion_matrix(test_labels, y_pred), classification_report(test_labels, y_pred))


def train_model_batch(model, config, test, test_labels, acctrain=False, resume=None):
    """
    Trains the model using Keras batch method

    :param resume:
    :param acctrain:
    :param model:
    :param config:
    :param test:
    :param test_labels:
    :return:
    """
    iepoch = 0
    if config['optimizer']['method'] == 'adagrad':
        optimizer = Adagrad()
    elif config['optimizer']['method'] == 'adadelta':
        optimizer = Adadelta()
    elif config['optimizer']['method'] == 'adam':
        optimizer = Adam()
    else:  # default SGD
        params = config['optimizer']['params']
        if resume is None:  # New experiment
            optimizer = SGD(lr=params['lrate'], momentum=params['momentum'], decay=params['decay'],
                            nesterov=params['nesterov'])
            iepoch = 0
        else:  # Resume training
            lrate = params['lrate'] - ((params['lrate'] / config['train']['epochs']) * params['epochs_trained'])

            optimizer = SGD(lr=lrate, momentum=params['momentum'], decay=params['decay'],
                            nesterov=params['nesterov'])
            iepoch = config['train']['epochs_trained']

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    classweight = detransweights(config['train']['classweight'])
    dblog = DBLog(database=mongoconnection, config=config, model=model, modelj=model.to_json(), resume=resume)

    ldaysTr = config['traindata']
    reb = config['rebalanced']

    # Train Epochs
    logs = {'loss': 0.0, 'acc': 0.0, 'val_loss': 0.0, 'val_acc': 0.0}
    for epoch in range(iepoch, config['train']['epochs']):
        shuffle(ldaysTr)
        tloss = []
        tacc = []

        # Train Batches
        for day in ldaysTr:
            X_train, y_train, perm = dayGenerator(config['datapath'], day, config['zfactor'], config['num_classes'],
                                                  config['train']['batchsize'], reb=reb, imgord=config['imgord'])
            for p in perm:
                loss = model.train_on_batch(X_train[p], y_train[p], class_weight=classweight)
                tloss.append(loss[0])
                tacc.append(loss[1])

        # If acctrain is true then test all the train with the retrained model to obtain the real loss and acc after training
        # los and accuracy during the training is not accurate, but, in the end, the real measure of generalization
        #  is obtained with the independent test
        if acctrain:
            tloss = []
            tacc = []
            # Test Batches
            for day in ldaysTr:
                X_train, y_train, perm = dayGenerator(config['datapath'], day, config['zfactor'], config['num_classes'],
                                                      config['train']['batchsize'], reb=reb, imgord=config['imgord'])
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

        if config['savepath']:
            model.save(config['savepath'] + '/' + str(dblog.id) + '.h5')

    scores = model.evaluate(test[0], test[1], verbose=0)
    dblog.on_train_end(logs={'acc': logs['acc'], 'val_acc': scores[1]})
    y_pred = model.predict_classes(test[0], verbose=0)
    dblog.save_final_results(scores, confusion_matrix(test_labels, y_pred), classification_report(test_labels, y_pred))


def load_dataset(config, only_test=False, imgord='th'):
    """
    Loads the train and test dataset

    :return:
    """
    ldaysTr = config['traindata']
    ldaysTs = config['testdata']
    z_factor = config['zfactor']
    datapath = config['datapath']

    if not only_test:
        X_train, y_trainO = load_generated_dataset(datapath, ldaysTr, z_factor)
        # Data already generated in theano order
        # if imgord == 'th':
        #     X_train = X_train.transpose((0,3,1,2))
        y_train = np_utils.to_categorical(y_trainO, len(np.unique(y_trainO)))
    else:
        X_train = None,
        y_train = None

    X_test, y_testO = load_generated_dataset(datapath, ldaysTs, z_factor)

    # Data already generated in theano order
    # if imgord == 'th':
    #     X_test = X_test.transpose((0,3,1,2))
    y_test = np_utils.to_categorical(y_testO, len(np.unique(y_testO)))

    num_classes = y_test.shape[1]

    return (X_train, y_train), (X_test, y_test), y_testO, num_classes


def load_generated_dataset(datapath, ldaysTr, z_factor):
    """
    Load the already generated datasets

    :param datapath:
    :param ldaysTr:
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
