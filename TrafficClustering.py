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
from sklearn.cluster import KMeans
from scipy.ndimage import zoom
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cbook as cbook
import matplotlib.image as mpimg

__author__ = 'bejar'

day = '20161029'

ldir = []
for c in ['Ronda*']:
    ldir.extend(glob.glob(cameras_path+day+'/*%s.gif'%c))

ldata = []

for dir in ldir:
    im = Image.open(dir).convert('RGB')
    data = np.asarray(im)
    data = np.dstack((zoom(data[:,:,0], 0.5), zoom(data[:,:,1], 0.5), zoom(data[:,:,2], 0.5)))
    data = np.reshape(data, (data.shape[0]*data.shape[1]*data.shape[2]))
    # print(data.shape)
    ldata.append(data)

adata = np.array(ldata)
pca = PCA(n_components=20)

pcadata = pca.fit_transform(adata)

v = 0
for i in range(15):
    v += pca.explained_variance_ratio_[i]
print(v)

# fig = plt.figure()
# fig.set_figwidth(30)
# fig.set_figheight(30)
#
# plt.scatter(pcadata[:,0], pcadata[:,1])
#
# plt.show()
# plt.close()

tsne = TSNE(n_components=2, n_iter=10000, perplexity=200.0, early_exaggeration=4.0, learning_rate=100.0)
tsdata = tsne.fit_transform(pcadata)
print(tsne.kl_divergence_ )

fig = plt.figure()
fig.set_figwidth(30)
fig.set_figheight(30)

# ax = fig.add_subplot(111, projection='3d')
# plt.scatter(tsdata[:,0], tsdata[:,1],zs=tsdata[:,2], c='r', s=30, depthshade=False, marker='o')


plt.scatter(tsdata[:,0], tsdata[:,1])

plt.show()
plt.close()

kmeans = KMeans(n_clusters=20, n_jobs=-1)

kmeans.fit_transform(tsdata)

fig = plt.figure()
fig.set_figwidth(30)
fig.set_figheight(30)
plt.scatter(tsdata[:,0], tsdata[:,1], c=kmeans.labels_)

plt.show()
plt.close()

for c in np.unique(kmeans.labels_):
    bools = kmeans.labels_ == c

    for i  in range(bools.shape[0]):
        if bools[i]:
            print(c, ldir[i])
            # im = Image.open(ldir[i])#.convert('RGB')
            # image =np.asarray(im)
            image = mpimg.imread(ldir[i])
            fig = plt.figure()
            fig.set_figwidth(30)
            fig.set_figheight(30)
            plt.imshow(image)
            plt.show()
            plt.close()



