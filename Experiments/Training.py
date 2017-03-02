'''
.. module:: Training

ConvoBatch
*************

  Trains a model according to a JSON configuration file
  Model is trained using the train_on_batch method from Keras model, so only a chunk of data is loaded in memory at a time
  Data is managed using the Dataset class

:Description: ConvoBatch

:Authors: bejar

:Version: 

:Created on: 23/12/2016 15:05 

'''

__author__ = 'bejar'

import argparse
import json

import keras.models
import numpy as np
from Traffic.Callback import DBLog, FileLog
from Traffic.Data.Dataset import Dataset
from Traffic.Models.SimpleModels import simple_model
from Traffic.Private.DBConfig import mongoconnection
from keras import backend as K
from keras.optimizers import SGD, Adagrad, Adadelta, Adam
from numpy.random import shuffle
from pymongo import MongoClient
from sklearn.metrics import confusion_matrix, classification_report
from Traffic.Util.Misc import load_config_file,  detransweights, recoding_dictionary

__author__ = 'bejar'


def train_model_batch_lrate_schedule(model, config, test):
    """
    Trains the model when there is a schedule of learning rates

    :param model:
    :param config:
    :return:
    """

    if 'log' not in config or config['log'] == 'db':
        dblog = DBLog(database=mongoconnection, config=config, model=model, modelj=model.to_json(), resume=resume)
    else:
        dblog = FileLog(config=config, modelj=model.to_json())

    classweight = detransweights(config['train']['classweight'])
    recode = None if 'recode' not in config else recoding_dictionary(config['recode'])
    train = Dataset(config['datapath'], config['traindata'], config['zfactor'], imgord=config['imgord'],
                    nclasses=test.nclasses, recode=recode)

    # For now only SDG
    for lrate, nepochs in zip(config['optimizer']['params']['lrate'], config['train']['epochs']):
        params = config['optimizer']['params']
        optimizer = SGD(lr=lrate, momentum=params['momentum'], decay=params['decay'],
                        nesterov=params['nesterov'])

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Train Epochs
        logs = {'loss': 0.0, 'acc': 0.0, 'val_loss': 0.0, 'val_acc': 0.0}

        for epoch in range(nepochs):
            train.open()
            chunks, _ = train.chunks()

            shuffle(chunks)

            # Train Batches
            lloss = []
            lacc = []
            for chunk in chunks:
                train.load_chunk(chunk, config['train']['batchsize'])

                for p in train.perm:
                    loss, acc = model.train_on_batch(train.X_train[p], train.y_train[p], class_weight=classweight)
                    lloss.append(loss)
                    lacc.append(acc)

            train.close()

            logs['loss'] = float(np.mean(lloss))
            logs['acc'] = float(np.mean(lacc))

            logs['val_loss'], logs['val_acc'] = model.evaluate(test.X_train, test.y_train, verbose=0)

            force_stop = dblog.force_stop()
            dblog.on_epoch_end(epoch, logs=logs)

            if config['savepath']:
                model.save(config['savepath'] + '/' + str(dblog.id) + '.h5')



            # If the training is stopped remotely training stops
            if force_stop:
                break
        if force_stop:
            break

    scores = model.evaluate(test.X_train, test.y_train, verbose=0)
    dblog.on_train_end(logs={'acc':logs['acc'], 'val_acc':scores[1]})
    y_pred = model.predict_classes(test.X_train, verbose=0)
    dblog.save_final_results(scores, confusion_matrix(test.y_labels, y_pred), classification_report(test.y_labels, y_pred))



def train_model_batch(model, config, test, resume=None):
    """
    Trains the model using Keras train batch method

    :param model:
    :param config:
    :param test:
    :param test_labels:
    :return:
    """


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
        else: # Resume training
            nlrate = params['lrate'] - ((params['lrate'] / config['train']['epochs']) * params['epochs_trained'])

            optimizer = SGD(lr=nlrate, momentum=params['momentum'], decay=params['decay'],
                            nesterov=params['nesterov'])
            iepoch = config['train']['epochs_trained']


    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    classweight = detransweights(config['train']['classweight'])
    if 'log' not in config or config['log'] == 'db':
        dblog = DBLog(database=mongoconnection, config=config, model=model, modelj=model.to_json(), resume=resume)
    else:
        dblog = FileLog(config=config, modelj=model.to_json())

    recode = None if 'recode' not in config else recoding_dictionary(config['recode'])

    train = Dataset(config['datapath'], config['traindata'], config['zfactor'], imgord=config['imgord'],
                    nclasses=test.nclasses, recode=recode)

    # Train Epochs
    logs = {'loss': 0.0, 'acc': 0.0, 'val_loss': 0.0, 'val_acc': 0.0}

    for epoch in range(iepoch, config['train']['epochs']):
        train.open()
        chunks, _ = train.chunks()
        shuffle(chunks)

        # Train Batches
        lloss = []
        lacc = []
        for chunk in chunks:
            train.load_chunk(chunk, config['train']['batchsize'])

            for p in train.perm:
                loss, acc = model.train_on_batch(train.X_train[p], train.y_train[p], class_weight=classweight)
                lloss.append(loss)
                lacc.append(acc)
        train.close()

        logs['loss'] = float(np.mean(lloss))
        logs['acc'] = float(np.mean(lacc))

        logs['val_loss'], logs['val_acc'] = model.evaluate(test.X_train, test.y_train, verbose=0)

        force_stop = dblog.force_stop()
        dblog.on_epoch_end(epoch, logs=logs)

        if config['savepath']:
            model.save(config['savepath'] + '/' + str(dblog.id) + '.h5')

        # If the training is stopped remotely training stops
        if force_stop:
            break

    scores = model.evaluate(test.X_train, test.y_train, verbose=0)
    dblog.on_train_end(logs={'acc':logs['acc'], 'val_acc':scores[1]})
    y_pred = model.predict_classes(test.X_train, verbose=0)
    dblog.save_final_results(scores, confusion_matrix(test.y_labels, y_pred), classification_report(test.y_labels, y_pred))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config', help='Experiment configuration')
    parser.add_argument('--resume', default=None, help='Resume existing experiment training')
    parser.add_argument('--retrain', default=None, help='Continue existing experiment training')
    args = parser.parse_args()

    config = load_config_file(args.config)

    K.set_image_dim_ordering(config['imgord'])

    # Only the test set in memory, the training is loaded in batches
    testdays = []
    recode = None if 'recode' not in config else recoding_dictionary(config['recode'])
    test = Dataset(config['datapath'], config['testdata'], config['zfactor'], imgord=config['imgord'], recode=recode)
    test.open()

    test.in_memory()

    config['input_shape'] = test.input_shape
    config['num_classes'] = test.nclasses

    resume = None
    if args.retrain is not None:  # Network already trained
        client = MongoClient(mongoconnection.server)
        db = client[mongoconnection.db]
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
        col = db[mongoconnection.col]

        vals = col.find_one({'_id': int(args.retrain)}, {'config':1})
        if vals is None:
            raise ValueError('This experiment does not exist ' + args.retrain)
        else:
            if config['zfactor'] != vals['config']['zfactor']:
                raise ValueError('Incompatible Data')
            weights = config['train']['classweight']
            for w in weights:
                if weights[w] != vals['config']['train']['classweight'][w]:
                    raise ValueError('Incompatible class weights')
            config['model'] = vals['config']['model']
            config['convolayers'] = vals['config']['convolayers']
            config['fulllayers'] = vals['config']['fulllayers']
            config['cont'] = args.retrain
            model = keras.models.load_model(config['savepath'] + args.retrain + '.h5')
    elif args.resume is not None: # Network interrupted
        client = MongoClient(mongoconnection.server)
        db = client[mongoconnection.db]
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
        col = db[mongoconnection.col]

        vals = col.find_one({'_id': int(args.retrain)}, {'config': 1, 'acc': 1})
        if vals is None:
            raise ValueError('This experiment does not exist ' + args.resume)
        else:
            config = vals['config']
            config['train']['epochs_trained'] = len(config['acc'])
            model = keras.models.load_model(config['savepath'] + args.resume + '.h5')
            resume = vals

    else:  # New model
        model = simple_model(config)

    if type(config['optimizer']['params']['lrate']) != list:
        train_model_batch(model, config, test, resume=resume)
    else:
        train_model_batch_lrate_schedule(model, config, test)
    test.close()
