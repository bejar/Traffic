"""
.. module:: ViewImages

ViewImages
*************

:Description: ViewImages

    

:Authors: bejar
    

:Version: 

:Created on: 16/11/2016 8:53 

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


imgpath = '/home/bejar/storage/Data/Traffic/Cameras/'
day = '20161029'
nclass = 6

ldir = glob.glob(imgpath+day+'/*.gif')

ldata = []

for dir in ldir:
    image = mpimg.imread(dir)
    im = Image.open(dir).convert('RGB')
    print(image.shape)
    image = image[5:235, 5:315,:]
    im = np.asarray(im)
    im = im[5:235, 5:315,:]
    thres = np.sum(image == 254)
    if thres > 1000:
        fig = plt.figure()
        fig.set_figwidth(60)
        fig.set_figheight(30)
        sp1 = fig.add_subplot(1,2,1)
        sp1.imshow(image[:,:,0])
        sp1 = fig.add_subplot(1,2,2)
        sp1.hist(image.ravel(), bins=256,  fc='k', ec='k')
        plt.show()
        plt.close()
