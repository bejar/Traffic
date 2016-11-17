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

from collections import Counter

import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from sklearn.decomposition import IncrementalPCA

from Util.Constants import cameras_path,data_path
from Util.Generate_Dataset import generate_classification_dataset

__author__ = 'bejar'


z_factor = 0.5

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

print(X_train.shape, X_test.shape)

np.save(data_path + 'train_data%0.2f.npy'%z_factor, X_train)
np.save(data_path + 'train_labels.npy', np.array(y_train))
np.save(data_path + 'test_data%0.2f.npy'%z_factor, X_test)
np.save(data_path + 'test_labels.npy', np.array(y_test))
