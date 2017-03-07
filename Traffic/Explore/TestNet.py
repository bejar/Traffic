"""
.. module:: TestNet

TestNet
*************

 Loads a learned model and visualizes the prediction for a test set

:Description: TestNet


:Authors: bejar
    

:Version: 

:Created on: 26/01/2017 7:30 

"""

import pickle

import keras.models
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from Traffic.Config.Constants import models_path, dataset_path, cameras_path
from Traffic.Util.ConvoTrain import load_generated_dataset
from Traffic.Util.DataGenerators import list_days_generator
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report

__author__ = 'bejar'


def view_image(im, lab, pred):
    """
    Shows the image of a camera for
    :return:
    """
    image = mpimg.imread(cameras_path + im + '.gif')
    fig = plt.figure()
    fig.set_figwidth(30)
    fig.set_figheight(30)
    sp1 = fig.add_subplot(1, 1, 1)
    sp1.imshow(image)
    plt.title(im + ' L=' + str(lab) + ' P= ' + str(pred))
    plt.show()
    plt.close()


def load_day_files(day, z_factor):
    """
    loads the names of the files corresponding to the examples of a day dataset
    :param z_factor:
    :param day:
    :return:
    """
    output = open(dataset_path + 'images-D%s-Z%0.2f.pkl' % (day, z_factor), 'rb')
    limages = pickle.load(output)
    output.close()

    return limages


if __name__ == '__main__':
    nclasses = 5
    modelid = '1484736764'
    z_factor = 0.25
    days = list_days_generator(2016, 12, 1, 1)
    X, yO = load_generated_dataset(dataset_path, days, z_factor)
    y = np_utils.to_categorical(yO, nclasses)
    limages = []
    for day in days:
        limages.extend(load_day_files(day, z_factor))

    model = keras.models.load_model(models_path + modelid + '.h5')

    scores = model.evaluate(X, y, verbose=0)

    print(scores)
    y_pred = model.predict_classes(X, verbose=0)

    print(confusion_matrix(yO, y_pred))
    print(classification_report(yO, y_pred))

    sel = np.array(yO) == 4
    sel2 = np.array(y_pred) == 4

    for s, s2, l, p, im in zip(sel, sel2, yO, y_pred, limages):
        if s and s2:
            view_image(im, l, p)
