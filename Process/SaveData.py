"""
.. module:: Classification

Classification
*************

:Description: Classification

    Classification experiment

:Authors: bejar
    

:Version: 

:Created on: 15/11/2016 11:12 

"""

import numpy as np
from Util.Constants import cameras_path,data_path
from Util.Generate_Dataset import generate_dataset

__author__ = 'bejar'



if __name__ == '__main__':
    z_factor = 0.25
    ldaysTr = ['20161107', '20161108', '20161109','20161110', '20161111', '20161114']
    ldaysTs = ['20161115']


    X_train, y_train, X_test, y_test = generate_dataset(ldaysTr, ldaysTs, z_factor, PCA=True, ncomp=350)

    np.save(data_path + 'train_data.npy', X_train)
    np.save(data_path + 'train_labels.npy', np.array(y_train))
    np.save(data_path + 'test_data.npy', X_test)
    np.save(data_path + 'test_labels.npy', np.array(y_test))
