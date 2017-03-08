"""
.. module:: TrImage

TrImage
*************

:Description: TrImage

    Class to read and process Traffic Images.
    For now only the images from Barcelona are considered for checking no service image

    The process is always:

    Create the object once

    for each image
        load the image
        check if it is correct
        transform the image crop+zoom (the object returns the data)


:Authors: bejar
    

:Version: 

:Created on: 15/02/2017 7:12 

"""

from scipy.ndimage import zoom, imread
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from Traffic.Config.Constants import info_path
import glob

__author__ = 'bejar'


class TrImage:
    def __init__(self):
        """
        Object to process camera images.
        It loads no services images

        :param pimage:
        """
        self.bcnnoserv = np.asarray(Image.open(info_path + 'BCNnoservice.gif'))
        self.correct = False
        self.data = None
        self.trans = False

    def load_image(self, pimage):
        """
        Loads an image from a file
        :param pimage:
        :return:
        """
        self.data = Image.open(pimage)
        self.trans = False

    def is_correct(self):
        """
        Returns of the image is correct or has enough quality

        Different checks can be done to the image, for now only the "service not available" is check in the init method

        :return:
        """
        # Do image checking

        if self.data is not None and not self.trans:
            self.correct = True
            # Checks if it is no service image for BCN
            self.correct = self.correct and not np.all(np.asarray(self.data) == self.bcnnoserv)
        else:
            raise Exception('Image already transformed')
        return self.correct

    def transform_image(self, z_factor, crop=(0, 0, 0, 0)):
        """
        Performs the transformation of the image

        The idea is to check if the image is correct before applying the transformation

        :param z_factor:
        :param crop:
        :return:
        """
        if self.data is not None and not self.trans:
            img = self.data.crop((crop[0], crop[2], self.data.size[0] - crop[1], self.data.size[1] - crop[3]))
            img = img.resize((int(z_factor * img.size[0]), int(z_factor * img.size[1])), PIL.Image.ANTIALIAS).convert(
                'RGB')
            self.data = np.asarray(img) / 255.0  # Normalize to [0-1] range
            self.trans = True
        else:
            raise Exception('Image already transformed')
        return self.data

    def get_data(self):
        """
        Returns the data from the image, if it is not transformed yet
        :return:
        """
        if self.data is not None and self.trans:
            return self.data
        else:
            raise Exception('Image not yet transformed')

    def data_augmentation(self):
        """
        Generates variarions of the original image, now does nothing

        Possibilities: horizontal flip, (zoom in + crop) parts of the image
        :return:
        """
        if self.data is not None and self.trans:
            flipped = np.fliplr(self.data)
        else:
            raise Exception('Image not yet transformed')

        return [flipped]

    def show(self):
        """
        Plots the data from the image
        :return:
        """
        fig = plt.figure()
        fig.set_figwidth(10)
        fig.set_figheight(10)
        sp1 = fig.add_subplot(1, 1, 1)
        sp1.imshow(self.data)
        plt.show()
        plt.close()

    def bad_pixels(self, nbins, percentage):
        """
        Returs if the pixels binarize in nbins represent more than a percentage of the image
        :param perc:
        :return:
        """
        if self.data is not None and self.trans:
            cutout = int(self.data.shape[0] * self.data.shape[1] * (percentage/100.0))
            # mprod = self.data[:, :, 0] * (self.data[:, :, 1] + 1) * (self.data[:, :, 0] + 2)
            mprod = self.data[:, :, 0] + 10 * self.data[:, :, 1] + 100 * self.data[:, :, 0]
            hist, bins = np.histogram(mprod.ravel(), bins=nbins)
            return np.max(hist) > cutout
        else:
            raise Exception('Image not yet transformed')


    def histogram(self):
        """
        plots the colors histograms of the rgb channels of the image

        :return:
        """
        if self.data is not None and self.trans:
            fig = plt.figure()
            fig.set_figwidth(10)
            fig.set_figheight(10)
            sp1 = fig.add_subplot(1, 2, 1)
            sp1.imshow(self.data)
            mprod = self.data[:, :, 0] + 10 * self.data[:, :, 1] + 100 * self.data[:, :, 0]
            hist, bins = np.histogram(mprod.ravel(), bins=50)
            sp2 = fig.add_subplot(1, 2, 2)
            sp2.plot(bins[:-1], hist, 'r')
            plt.show()
            plt.close()
        else:
            raise Exception('Image not yet transformed')

if __name__ == '__main__':
    from Traffic.Config.Constants import cameras_path
    limg = sorted(glob.glob(cameras_path + '/20161101/*.gif'))
    image = TrImage()
    for img in limg:
        image.load_image(img)
        image.transform_image(z_factor=0.5, crop=(5, 5, 5, 5))
        if image.bad_pixels(100, 20):
            image.histogram()

    # image.load_image(cameras_path + '/20161101/201611011453-RondaLitoralZonaFranca.gif')
    # image.load_image(cameras_path + '/20161101/201611010004-PlPauVila.gif')
    # image.show()
    # image.transform_image(z_factor=0.5, crop=(5, 5, 5, 5))
    # image.histogram()
    # if image.is_correct():
    #     im = image.transform_image(z_factor=0.5, crop=(5, 5, 5, 5))
    #     print im.shape
    #     image.show()
    #
    #     nimg = TrImage()
    #     nimg.data = image.data_augmentation()[0]
    #     nimg.show()
    #
    # image.load_image(cameras_path + '/20161101/201611010004-PlPauVila.gif')
    # image.show()
    # if image.is_correct():
    #     im = image.transform_image(z_factor=0.5, crop=(5, 5, 5, 5))
    #     print im.shape
    #     image.show()
    # else:
    #     print('Incorrect Image')
