"""
.. module:: ViewImagesMAD

ViewImagesMAD
*************

:Description: ViewImagesMAD

    

:Authors: bejar
    

:Version: 

:Created on: 21/11/2016 13:31 

"""

import glob

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from PIL import ImageOps
import csv

from Util.Constants import cameras_path_MAD, data_path_MAD
from tesserocr import PyTessBaseAPI
import tesserocr
__author__ = 'bejar'



day = '20161117'

ldir = glob.glob(cameras_path_MAD+day+'/*LOPEZ*.jpg')

for im in sorted(ldir):
    image = Image.open(im)
    w, h = image.size
    for i in range(0, 40, 5):
        image2 = image.crop((247+i, 212, 247+(i+5), h))
        fig = plt.figure()
        fig.set_figwidth(30)
        fig.set_figheight(30)
        sp1 = fig.add_subplot(1, 1, 1)
        sp1.imshow(image2)
        # sp1.imshow(image[210:,220:])
        plt.title(im)
        plt.show()
        plt.close()

# with PyTessBaseAPI() as api:
#     for im in sorted(ldir):
#         image = Image.open(im)
#         w, h = image.size
#         image = image.crop((210, 212, w, h))
#         # image.show()
#         # enhancer = ImageEnhance.Sharpness(image)
#         # image = enhancer.enhance(2)
#         # image = image.filter(ImageFilter.SHARPEN)
#         # image = image.filter(ImageFilter.SHARPEN)
#         # image = image.filter(ImageFilter.SHARPEN)
#         # image = ImageOps.grayscale(image)
#
#         # print(tesserocr.image_to_text(image))
#         # api.SetImageFile(im)
#         # print(api.GetUTF8Text())
#         # print(api.AllWordConfidences())
#         # image = mpimg.imread(im)
#
#

