"""
.. module:: Generate_Dataset

Generate_Dataset
*************

:Description: Generate_Dataset

    Generate a dataset for traffic images

:Authors: bejar
    

:Version: 

:Created on: 14/11/2016 12:53 

"""

import glob

import numpy as np

from Process.CamTram import CamTram
from Util.Constants import cameras_path, status_path
from Util.DataTram import DataTram
from collections import Counter

import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from sklearn.decomposition import IncrementalPCA

from Util.Constants import cameras_path,data_path

__author__ = 'bejar'


def generate_classification_dataset(day):
    """
    Generates a dictionary with the dates of the images with lists that contain the camera name and current and predicted
    traffic status

    :param day:
    :return:
    """
    #day = '20161031'

    CTram = CamTram()

    ldir = glob.glob(cameras_path + day + '/*.gif')

    camdic = {}

    for f in sorted(ldir):
        name = f.split('.')[0].split('/')[-1]
        time, place = name.split('-')
        # print(time, place, CTram.ct[place])
        if int(time) in camdic:
            camdic[int(time)].append(place)
        else:
            camdic[int(time)] = [place]

    # print(camdic)

    ldir = glob.glob(status_path + day + '/*-dadestram.data')
    ldata = []
    for f in sorted(ldir):
        ldata.append(DataTram(f))


    assoc = {}

    for imgtime in sorted(camdic):
        # Look for the status and forecast closer to the image but always in the future
        dmin = None
        dmin2 = None
        vmin = 10000
        for d in ldata:
            if vmin > np.abs(imgtime - d.date): # Only if it is ahead in time
                if imgtime - d.date >0:
                    vmin = np.abs(imgtime - d.date)
                    dmin = d
                    dmin2 = dmin
        if dmin is not None and dmin2 is not None:
            lclass = []
            for img in camdic[imgtime]:
                tram = CTram.ct[img][0]
                # print(imgtime, dmin.dt[tram], img)
                # store for an image of that time the name, closest status, prediction and next status
                lclass.append((img, dmin.dt[tram][0], dmin.dt[tram][1], dmin2.dt[tram][0]))
            assoc[imgtime] = lclass

    return assoc


def generate_dataset(ldaysTr, ldaysTs, z_factor, PCA=True, ncomp=100):
    """
    Generates a training and test datasets from the days in the parameters
    z_factor is the zoom factor to rescale the images
    :param trdays:
    :param tsdays:
    :return:
    """

    # -------------------- Train Set ------------------
    ldataTr = []
    llabelsTr = []

    for day in ldaysTr:
        dataset = generate_classification_dataset(day)
        for t in dataset:
            for cam, l, _, _ in dataset[t]:
                # print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                if l != 0 and l != 6:
                    image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                    if np.sum(image == 254) < 100000:
                        del image
                        im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
                        data = np.asarray(im)
                        data = data[5:235, 5:315, :].astype('float32')
                        data /= 255.0
                        if z_factor is not None:
                            data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
                                              zoom(data[:, :, 2], z_factor)))
                        data = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2]))
                        ldataTr.append(data)
                        llabelsTr.append(l)

    # ------------- Test Set ------------------

    ldataTs = []
    llabelsTs = []

    for day in ldaysTs:
        dataset = generate_classification_dataset(day)
        for t in dataset:
            for cam, l, _, _ in dataset[t]:
                # print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                if l != 0 and l != 6:
                    image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                    if np.sum(image == 254) < 100000:
                        del image
                        im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
                        data = np.asarray(im)
                        data = data[5:235, 5:315, :].astype('float32')
                        data /= 255.0
                        if z_factor is not None:
                            data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
                                              zoom(data[:, :, 2], z_factor)))
                        data = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2]))
                        ldataTs.append(data)
                        llabelsTs.append(l)
    del data

    print(Counter(llabelsTr))
    print(Counter(llabelsTs))

    X_train = np.array(ldataTr)
    del ldataTr
    X_test = np.array(ldataTs)
    del ldataTs

    if PCA:
        pca = IncrementalPCA(n_components=ncomp)
        pca.fit(X_train)
        print(np.sum(pca.explained_variance_ratio_[:ncomp]))
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    y_train = llabelsTr
    y_test = llabelsTs
    print(X_train.shape, X_test.shape)

    return X_train, y_train, X_test, y_test


def generate_daily_dataset(ldaysTr, ldaysTs, z_factor, PCA=True, ncomp=100):
    """
    Comptes the PCA transformation using the days in ldaysTr
    Generates  datasets from the days in the ldaysTs
    z_factor is the zoom factor to rescale the images
    :param trdays:
    :param tsdays:
    :return:
    """

    # -------------------- Train Set ------------------
    ldataTr = []
    llabelsTr = []

    for day in ldaysTr:
        dataset = generate_classification_dataset(day)
        for t in dataset:
            for cam, l, _, _ in dataset[t]:
                # print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                if l != 0 and l != 6:
                    image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                    if np.sum(image == 254) < 100000:
                        del image
                        im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
                        data = np.asarray(im)
                        data = data[5:235, 5:315, :].astype('float32')
                        data /= 255.0
                        if z_factor is not None:
                            data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
                                              zoom(data[:, :, 2], z_factor)))
                        data = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2]))
                        ldataTr.append(data)
                        llabelsTr.append(l)

    print(Counter(llabelsTr))
    X_train = np.array(ldataTr)
    pca = IncrementalPCA(n_components=ncomp)
    pca.fit(X_train)
    print(np.sum(pca.explained_variance_ratio_[:ncomp]))
    del X_train

    # ------------- Test Set ------------------


    for day in ldaysTs:
        ldataTs = []
        llabelsTs = []
        dataset = generate_classification_dataset(day)
        for t in dataset:
            for cam, l, _, _ in dataset[t]:
                # print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                if l != 0 and l != 6:
                    image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                    if np.sum(image == 254) < 100000:
                        del image
                        im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
                        data = np.asarray(im)
                        data = data[5:235, 5:315, :].astype('float32')
                        data /= 255.0
                        if z_factor is not None:
                            data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
                                              zoom(data[:, :, 2], z_factor)))
                        data = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2]))
                        ldataTs.append(data)
                        llabelsTs.append(l)
        X_test = pca.transform(np.array(ldataTs))
        y_test = llabelsTs
        print(Counter(llabelsTs))
        np.save(data_path + 'data-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp), X_test)
        np.save(data_path + 'labels-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp), np.array(y_test))
