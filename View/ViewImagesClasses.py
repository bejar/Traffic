"""
.. module:: ViewImagesClasses

ViewImagesClasses
*************

:Description: ViewImagesClasses

    

:Authors: bejar
    

:Version: 

:Created on: 16/11/2016 13:40 

"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from Util.Constants import cameras_path
from Util.Generate_Dataset import generate_classification_dataset

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
