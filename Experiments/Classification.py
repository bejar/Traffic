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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from Util.Generate_Dataset import generate_dataset
from Util.Constants import data_path

__author__ = 'bejar'

if __name__ == '__main__':

    pre = True
    if pre:
       X_train = np.load(data_path + 'train_data.npy')
       y_train = np.load(data_path + 'train_labels.npy')
       X_test = np.load(data_path + 'test_data.npy')
       y_test = np.load(data_path + 'test_labels.npy')
    else:
        z_factor = 0.25
        ldaysTr = ['20161107', '20161108', '20161109','20161110', '20161111', '20161114']
        ldaysTs = ['20161115']
        X_train, y_train, X_test, y_test = generate_dataset(ldaysTr, ldaysTs, z_factor, PCA=True, ncomp=350)

    clsf = 'SVM'

    if clsf == 'GB':
        for est, depth in product([300, 400, 500, 600], [3, 5, 7]):
            print('Estimators= %d Depth= %d Time= %s' %(est, depth, time.ctime()))
            gb = GradientBoostingClassifier(n_estimators=est, max_depth=depth)
            scores = cross_val_score(gb, X_train, y_train, cv=10)
            print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

            gb.fit(X_train, y_train)
            labels = gb.predict(X_test)
            print('Test Accuracy: %0.2f'% gb.score(X_test, y_test))
            print(confusion_matrix(y_test, labels, labels=sorted(np.unique(y_test))))
    elif clsf == 'SVM':
        for C in [1.1, 1.5]:
            print('C= %f Time= %s' %(C, time.ctime()))
            svm = SVC(C=C, kernel='poly', degree=3, coef0=1, class_weight='balanced')

            scores = cross_val_score(svm, X_train, y_train, cv=10)
            print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

            svm.fit(X_train, y_train)
            labels = svm.predict(X_test)
            print('Test Accuracy: %0.2f'% svm.score(X_test, y_test))
            print(confusion_matrix(y_test, labels, labels=sorted(np.unique(y_test))))
