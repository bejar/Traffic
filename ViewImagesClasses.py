"""
.. module:: ViewImagesClasses

ViewImagesClasses
*************

:Description: ViewImagesClasses

    

:Authors: bejar
    

:Version: 

:Created on: 16/11/2016 13:40 

"""

from PIL import Image
from Generate_Dataset import generate_classification_dataset
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


ldaysTr = ['20161108']

for day in ldaysTr:
    dataset = generate_classification_dataset(day)
    for t in sorted(dataset):
        for cam, l, _, _ in sorted(dataset[t]):
            # print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
            if l == 4:
                image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                fig = plt.figure()
                fig.set_figwidth(30)
                fig.set_figheight(30)
                sp1 = fig.add_subplot(1,1,1)
                sp1.imshow(image)
                plt.title(str(t) + ' ' + cam)
                plt.show()
                plt.close()
