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

from Generate_Dataset import generate_classification_dataset
from PIL import Image
import numpy as np
import glob
from Constants import cameras_path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.ndimage import zoom
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cbook as cbook
import matplotlib.image as mpimg
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


__author__ = 'bejar'


ldata = []
llabels = []

#ldays = ['20161031', '20161102', '20161103', '20161104']

ldays = ['20161102']

for day in ldays:
    dataset = generate_classification_dataset(day)
    for t in dataset:
        for cam, l, _ in dataset[t]:
            # print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
            im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
            data = np.asarray(im)
            data = np.dstack((zoom(data[:,:,0], 0.5), zoom(data[:,:,1], 0.5), zoom(data[:,:,2], 0.5)))
            data = np.reshape(data, (data.shape[0]*data.shape[1]*data.shape[2]))
            ldata.append(data)
            llabels.append(l)

print(Counter(llabels))
adata = np.array(ldata)

ncomp = 30
pca = PCA(n_components=ncomp)

pcadata = pca.fit_transform(adata)

v = 0
for i in range(ncomp):
    v += pca.explained_variance_ratio_[i]
print(v)
#
# fig = plt.figure()
# fig.set_figwidth(30)
# fig.set_figheight(30)
#
# plt.scatter(pcadata[:,0], pcadata[:,1], c=llabels)
#
# plt.show()
# plt.close()


X_train, X_test, y_train, y_test = train_test_split(pcadata, llabels, test_size=0.33, stratify=llabels)

# gb.fit(X_train, y_train)
# print(gb.score(X_test, y_test))

gb = GradientBoostingClassifier(n_estimators=200, max_depth=4)
scores = cross_val_score(gb, X_train, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

labels = gb.score(X_test, y_test)

print(confusion_matrix(y_test, labels, labels=sorted(np.unique(y_test))))