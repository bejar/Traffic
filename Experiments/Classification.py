"""
.. module:: Classification

Classification
*************

:Description: Classification

    Classification experiment

:Authors: bejar
    

:Version: 

:Created on: 15/11/2016 11:12 

"""

import time
from itertools import product
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from Util.Generate_Dataset import generate_dataset
from Util.Constants import data_path, dataset_path
from Util.Logger import config_logger

__author__ = 'bejar'

if __name__ == '__main__':

    log = config_logger(file='classification-' + time.strftime('%Y%m%d%H%M%S', time.localtime(int(time.time()))))
    #ldaysTr = ['20161101', '20161102', '20161103', '20161104', '20161105', '20161106', '20161107', '20161117', '20161118', '20161119', '20161120']
    ldaysTr = ['20161101', '20161102', '20161103', '20161104', '20161105', '20161106', '20161107', '20161108',
               '20161109', '20161110', '20161111', '20161112', '20161113', '20161114', '20161115', ]
    #ldaysTr = ['20161107', '20161108', '20161109', '20161110', '20161111', '20161112', '20161113']
    ldaysTs = ['20161116', '20161116', '20161117', '20161118']
    z_factor = 0.25
    ncomp = 1000
    PCA = True

    dataset = 'pre'  # 'pre' rb' 'gen'
    if dataset == 'pre':
        log.info('Train= %s  Test= %s z_factor= %0.2f PCA= %s NCOMP= %d', ldaysTr, ldaysTs, z_factor, PCA, ncomp)
        ldata = []
        y_train = []
        for day in ldaysTr:
            data = np.load(dataset_path + 'data-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp))
            ldata.append(data)
            y_train.extend(np.load(dataset_path + 'labels-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp)))
        X_train = np.concatenate(ldata)
        print(X_train.shape)

        ldata = []
        y_test = []
        for day in ldaysTs:
            data = np.load(dataset_path + 'data-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp))
            ldata.append(data)
            y_test.extend(np.load(dataset_path + 'labels-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp)))
        X_test = np.concatenate(ldata)
        print(X_test.shape)
        del ldata
    elif dataset == 'rb':
        log.info('Train= RB  Test= %s z_factor= %0.2f PCA= %s NCOMP= %d', ldaysTs, z_factor, PCA, ncomp)
        X_train = np.load(data_path + 'data-RB-Z%0.2f-C%d.npy' % (z_factor, ncomp))
        y_train = np.load(data_path + 'labels-RB-Z%0.2f-C%d.npy' % (z_factor, ncomp))
        print(X_train.shape)

        ldata = []
        y_test = []
        for day in ldaysTs:
            data = np.load(dataset_path + 'data-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp))
            ldata.append(data)
            y_test.extend(np.load(dataset_path + 'labels-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp)))
        X_test = np.concatenate(ldata)
        print(X_test.shape)
        del ldata
    else:
        X_train, y_train, X_test, y_test = generate_dataset(ldaysTr, ldaysTs, z_factor, PCA=PCA, ncomp=ncomp, method='two')
        log.info('Train= %s  Test= %s z_factor= %0.2f PCA= %s NCOMP= %d', ldaysTr, ldaysTs, z_factor, PCA, ncomp)

    clsf = 'SVM'

    if clsf == 'GB':
        log.info(' -- GB ----------------------')
        for est, depth in product([300, 400, 500, 600], [3, 5, 7]):
            log.info('Estimators= %d Depth= %d Time= %s', est, depth, time.ctime())
            gb = GradientBoostingClassifier(n_estimators=est, max_depth=depth)
            scores = cross_val_score(gb, X_train, y_train, cv=10)
            log.info("CV Accuracy: %0.2f (+/- %0.2f)", scores.mean(), scores.std() * 2)

            gb.fit(X_train, y_train)
            labels = gb.predict(X_test)
            log.info('Test Accuracy: %0.2f', gb.score(X_test, y_test))
            log.info('%s', str(confusion_matrix(y_test, labels, labels=sorted(np.unique(y_test)))))
            log.info('%s', classification_report(y_test, labels, labels=sorted(np.unique(y_test))))
    elif clsf == 'SVM':
        log.info(' -- SVM ----------------------')
        for C in [10, 100, 1000]:
            log.info('C= %f Time= %s', C, time.ctime())
            # svm = SVC(C=C, kernel='poly', degree=3, coef0=1, class_weight='balanced')
            svm = SVC(C=C, kernel='rbf', coef0=1, class_weight='balanced')

            scores = cross_val_score(svm, X_train, y_train, cv=10)

            log.info("CV Accuracy: %0.2f (+/- %0.2f)", scores.mean(), scores.std() * 2)

            svm.fit(X_train, y_train)

            labels = svm.predict(X_train)
            log.info('Train Accuracy: %0.2f', svm.score(X_train, y_train))
            log.info('%s', str(confusion_matrix(y_train, labels, labels=sorted(np.unique(y_train)))))
            log.info('%s', classification_report(y_train, labels, labels=sorted(np.unique(y_train))))

            labels = svm.predict(X_test)
            log.info('Test Accuracy: %0.2f', svm.score(X_test, y_test))
            log.info('%s', str(confusion_matrix(y_test, labels, labels=sorted(np.unique(y_test)))))
            log.info('%s', classification_report(y_test, labels, labels=sorted(np.unique(y_test))))

