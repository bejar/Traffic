"""
.. module:: TrafficClustering

TrafficClustering
*************

:Description: TrafficClustering

    

:Authors: bejar
    

:Version: 

:Created on: 15/11/2016 8:57 

"""

from PIL import Image
import numpy as np
import glob
from Constants import cameras_path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.ndimage import zoom
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

__author__ = 'bejar'

day = '20161029'


ldir = glob.glob(cameras_path+day+'/*Ronda*.gif')

ldata = []

for dir in ldir:
    im = Image.open(dir).convert('RGB')
    data = np.asarray(im)
    data = np.dstack((zoom(data[:,:,0], 0.5), zoom(data[:,:,1], 0.5), zoom(data[:,:,2], 0.5)))
    data = np.reshape(data, (data.shape[0]*data.shape[1]*data.shape[2]))
    # print(data.shape)
    ldata.append(data)

adata = np.array(ldata)
pca = PCA(n_components=100)

pcadata = pca.fit_transform(adata)

tsne = TSNE(n_components=3, perplexity=10.0, early_exaggeration=10.0, learning_rate=1000.0)

tsdata = tsne.fit_transform(pcadata)

fig = plt.figure()
fig.set_figwidth(30)
fig.set_figheight(30)

ax = fig.add_subplot(111, projection='3d')
plt.scatter(tsdata[:,0], tsdata[:,1],zs=tsdata[:,2], c='r', s=30, depthshade=False, marker='o')


# plt.scatter(tsdata[:,0], tsdata[:,1])

plt.show()
plt.close()
