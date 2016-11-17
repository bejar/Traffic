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
from collections import Counter
from itertools import product

import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from Util.Constants import cameras_path
from Util.Generate_Dataset import generate_classification_dataset

__author__ = 'bejar'


z_factor = 0.25

# -------------------- Train Set ------------------

ldataTr = []
llabelsTr = []

#ldays = ['20161031', '20161102', '20161103', '20161104']
ldays = ['20161107', '20161108', '20161109', '20161110', '20161111']

ldaysTr = ['20161107', '20161108', '20161109','20161110', '20161111', '20161114']

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
                    data = data[5:235, 5:315,:].astype('float32')
                    data /= 255.0
                    if z_factor is not None:
                        data = np.dstack((zoom(data[:,:,0], z_factor), zoom(data[:,:,1], z_factor), zoom(data[:,:,2], z_factor)))
                    data = np.reshape(data, (data.shape[0]*data.shape[1]*data.shape[2]))
                    ldataTr.append(data)
                    llabelsTr.append(l)

ldataTs = []
llabelsTs = []


# ------------- Test Set ------------------
ldaysTs = ['20161115']

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
                    data = data[5:235, 5:315,:].astype('float32')
                    data /= 255.0
                    if z_factor is not None:
                        data = np.dstack((zoom(data[:,:,0], z_factor), zoom(data[:,:,1], z_factor), zoom(data[:,:,2], z_factor)))
                    data = np.reshape(data, (data.shape[0]*data.shape[1]*data.shape[2]))
                    ldataTs.append(data)
                    llabelsTs.append(l)
del data

print(Counter(llabelsTr))
adata = np.array(ldataTr) #.extend(ldataTs))

ncomp = 350
pca = IncrementalPCA(n_components=ncomp)
pca.fit(adata)
print(np.sum(pca.explained_variance_ratio_[:ncomp]))

X_train = pca.transform(np.array(ldataTr))
y_train = llabelsTr
del ldataTr
X_test = pca.transform(np.array(ldataTs))
y_test = llabelsTs
del ldataTs

print(Counter(y_test))

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
    for C in [0.6, 0.7, 0.8, 0.9, 2]:
        print('C= %f Time= %s' %(C, time.ctime()))
        svm = SVC(C=C, kernel='poly', degree=3, coef0=1, class_weight='balanced')

        scores = cross_val_score(svm, X_train, y_train, cv=10)
        print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        svm.fit(X_train, y_train)
        labels = svm.predict(X_test)
        print('Test Accuracy: %0.2f'% svm.score(X_test, y_test))
        print(confusion_matrix(y_test, labels, labels=sorted(np.unique(y_test))))
