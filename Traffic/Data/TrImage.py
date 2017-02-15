"""
.. module:: TrImage

TrImage
*************

:Description: TrImage

    Class to read and process Traffic Images

    For now only the images from Barcelona are considered

:Authors: bejar
    

:Version: 

:Created on: 15/02/2017 7:12 

"""


from scipy.ndimage import zoom, imread
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image

__author__ = 'bejar'


class TrImage():

    def __init__(self, pimage, z_factor, crop=(0,0,0,0)):
        """
        Loads an image, checks if it is not the "service not available" image, applies the image crop and computes the soomed image

        :param pimage:
        """

        img = Image.open(pimage)
        img = img.crop((crop[0], crop[2], img.size[0]-crop[1], img.size[1]-crop[3]))

        if np.sum(np.asarray(img) == 254) < 100000:  # is not the "Service not available" image (only works for Barcelona)
            img = img.resize((int(z_factor * img.size[0]), int(z_factor * img.size[1])), PIL.Image.ANTIALIAS).convert('RGB')
            self.data = np.asarray(img)/255.0  # Normalize to [0-1] range
        else:
            self.correct = False
            self.data = None

    def correct(self):
        """
        Returns of the image is correct or has enough quality

        Different checks can be done to the image, for now only the "service not available" is check in the init method

        :return:
        """
        # Do image checking

        return self.correct

    def getData(self):
        """
        Returns the image as a
        :return:
        """
        return self.data

    def dataAugmentation(self):
        """
        Generates variarions of the original image, now does nothing

        Possibilities: horizontal flip, (zoom in + crop) parts of the image
        :return:
        """

        return None

    def show(self):
        """
        Plots the data from the image
        :return:
        """
        fig = plt.figure()
        fig.set_figwidth(10)
        fig.set_figheight(10)
        sp1 = fig.add_subplot(1,1,1)
        sp1.imshow(self.data)
        plt.show()
        plt.close()

if __name__ == '__main__':
    from Utilities.Constants import cameras_path
    image = TrImage(cameras_path + '/20161101/201611011453-RondaLitoralZonaFranca.gif', z_factor=0.25, crop=(5,5,5,5))
    print image.data.shape
    image.show()
