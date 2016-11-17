"""
.. module:: TFtest

TFtest
*************

:Description: TFtest

    

:Authors: bejar
    

:Version: 

:Created on: 04/11/2016 16:34 

"""

from keras.models import Sequential
from keras.layers import Convolution2D, Dense
from PIL import Image
import numpy
import glob
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

__author__ = 'bejar'

imgpath = '/home/bejar/storage/Data/Traffic/Cameras/'
day = '20161029'
nclass = 6

ldir = glob.glob(imgpath+day+'/*.gif')

ldata = []

for dir in ldir:
    im = Image.open(dir).convert('RGB')
    print(dir, im.format, im.size, im.mode)
    data = numpy.asarray(im)
    ldata.append(data)
    print(data.shape)
    fig = plt.figure()
    fig.set_figwidth(90)
    fig.set_figheight(30)
    sp1 = fig.add_subplot(1,3,1)
    sp1.imshow(data[:,:,0], cmap="Greys")
    sp1 = fig.add_subplot(1,3,2)
    sp1.imshow(data[:,:,1], cmap="Greys")
    sp1 = fig.add_subplot(1,3,3)
    sp1.imshow(data[:,:,2], cmap="Greys")
    plt.show()
    plt.close()

    dataz = zoom(data, 0.5)
    fig = plt.figure()
    fig.set_figwidth(90)
    fig.set_figheight(30)
    sp1 = fig.add_subplot(1,1,1)
    sp1.imshow(dataz[:,:,0], cmap="Greys")
    plt.hist(lum_img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.show()
    plt.close()



# model = Sequential()
#
# model.add(Convolution2D(64,3,3,input_shape=(ldata[0].shape)))
# model.add(Dense(nclass, activation='softmax'))
#



