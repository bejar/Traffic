"""
.. module:: ViewImages

ViewImages
*************

:Description: ViewImages

    

:Authors: bejar
    

:Version: 

:Created on: 16/11/2016 8:53 

"""

import glob

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from Util.Constants import cameras_path
from scipy.ndimage import zoom

__author__ = 'bejar'


day = '20161029'
nclass = 6

ldir = glob.glob(cameras_path+day+'/*.gif')

ldata = []
z_factor = 0.25
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
        sp1.imshow(zoom(image[:, :, 0], z_factor))
        # sp1.hist(image.ravel(), bins=256,  fc='k', ec='k')
        plt.show()
        plt.close()
