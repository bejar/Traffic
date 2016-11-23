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

from Util.Constants import cameras_path,data_path, dataset_path

__author__ = 'bejar'


def generate_classification_dataset_one(day):
    """
    Generates a dictionary with the dates of the images with lists that contain the camera name and current and predicted
    traffic status using only the nearest prediction in space and time

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
        vmin2 = 70
        vmin = 60
        for d in ldata:
            diff = dist_time(imgtime, d.date)
            if vmin > np.abs(diff): # Only if it is ahead in time
                if imgtime - d.date > 0:
                    vmin2 = vmin
                    vmin = diff
                    dmin2 = dmin
                    dmin = d
            elif vmin2 > np.abs(diff):
                if diff > 0 and diff != vmin:
                    vmin2 = diff
                    dmin2 = d

        if dmin is not None and dmin2 is not None and vmin < 60 and vmin2 < 60:
            lclass = []
            for img in camdic[imgtime]:
                tram = CTram.ct[img][0]
                # print(imgtime, dmin.dt[tram], img)
                # store for an image of that time the name, closest status, prediction and next status
                lclass.append((img, dmin.dt[tram][0], dmin.dt[tram][1], dmin2.dt[tram][0]))
            assoc[imgtime] = lclass

    return assoc


def dist_time(time1, time2):
    """
    distance between two hours

    :param time1:
    :param time2:
    :return:
    """

    t1 = (time1 % 100) + (60 * ((time1 // 100) % 100))
    t2 = (time2 % 100) + (60 * ((time2 // 100) % 100))
#    print(time1, time2, t1, t2, t2 - t1)
    return t2 - t1


def generate_classification_dataset_two(day):
    """
    Generates a dictionary with the dates of the images with lists that contain the camera name and current and predicted
    traffic status using the two nearest prediction in space and time

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
        vmin = 60
        vmin2 = 70
        for d in ldata:
            diff = dist_time(imgtime, d.date)
            if vmin > np.abs(diff): # Only if it is ahead in time
                if diff > 0:
                    vmin2 = vmin
                    vmin = diff
                    dmin2 = dmin
                    dmin = d
            elif vmin2 > np.abs(diff):
                if diff > 0 and diff != vmin:
                    vmin2 = diff
                    dmin2 = d

        if dmin is not None and dmin2 is not None and vmin < 60 and vmin2 < 60:
            lclass = []
            # print(imgtime, dmin.date, dmin2.date, vmin, vmin2)
            for img in camdic[imgtime]:
                tram1 = CTram.ct[img][0]
                tram2 = CTram.ct[img][1]
                # print(imgtime, dmin.dt[tram], img)
                # store for an image of that time the name, closest status, prediction and next status
                lclass.append((img,
                               max(dmin.dt[tram1][0], dmin.dt[tram2][0]),
                               max(dmin.dt[tram1][1], dmin.dt[tram2][1]),
                               max(dmin2.dt[tram1][0], dmin2.dt[tram2][0])))
            assoc[imgtime] = lclass

    return assoc


def generate_dataset(ldaysTr, ldaysTs, z_factor, PCA=True, ncomp=100, method='one'):
    """
    Generates a training and test datasets from the days in the parameters
    z_factor is the zoom factor to rescale the images
    :param ldaysTr:
    :param ldaysTs:
    :param z_factor:
    :param PCA:
    :param method:
    :return:
    :param ncomp:
    :return:
    """

    # -------------------- Train Set ------------------
    ldataTr = []
    llabelsTr = []

    for day in ldaysTr:
        if method == 'one':
            dataset = generate_classification_dataset_one(day)
        else:
            dataset = generate_classification_dataset_two(day)
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
        if method == 'one':
            dataset = generate_classification_dataset_one(day)
        else:
            dataset = generate_classification_dataset_two(day)
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


def generate_daily_dataset(ldaysTr, ldaysTs, z_factor, PCA=True, ncomp=100, method='one'):
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
        if method == 'one':
            dataset = generate_classification_dataset_one(day)
        else:
            dataset = generate_classification_dataset_two(day)
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
        if method == 'one':
            dataset = generate_classification_dataset_one(day)
        else:
            dataset = generate_classification_dataset_two(day)
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
        np.save(dataset_path + 'data-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp), X_test)
        np.save(dataset_path + 'labels-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp), np.array(y_test))


def generate_rebalanced_dataset(ldaysTr, ndays, z_factor, PCA=True, ncomp=100):
    """
    Generates a training dataset with a rebalance of the classes using a specific number of days of
    the input files for the training dataset

    :param ldaysTr:
    :param z_factor:
    :param PCA:
    :param ncomp:
    :return:
    """
    ldata = []
    y_train = []

    for cl, nd in ndays:
        for i in range(nd):
            day = ldaysTr[i]
            data = np.load(dataset_path + 'data-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp))
            labels = np.load(dataset_path + 'labels-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp))
            ldata.append(data[labels==cl,:])
            y_train.extend(labels[labels==cl])
    X_train = np.concatenate(ldata)
    print(X_train.shape)
    print(Counter(y_train))
    np.save(dataset_path + 'data-RB-Z%0.2f-C%d.npy' % (z_factor, ncomp), X_train)
    np.save(dataset_path + 'labels-RB-Z%0.2f-C%d.npy' % (z_factor, ncomp), np.array(y_train))



if __name__ == '__main__':
    generate_classification_dataset_two('20161101')