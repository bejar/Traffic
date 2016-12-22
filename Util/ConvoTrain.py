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
K.set_image_dim_ordering('th')
from Util.Generate_Dataset import generate_dataset, load_generated_dataset
from sklearn.metrics import confusion_matrix, classification_report
from Util.DBLog import DBLog
from Util.DBConfig import mongoconnection

__author__ = 'bejar'


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
                   callbacks=[dblog], class_weight=classweight, verbose=1)


    scores = model.evaluate(test[0], test[1], verbose=0)
    y_pred = model.predict_classes(test[0])

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
