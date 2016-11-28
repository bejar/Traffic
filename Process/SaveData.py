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
from Util.Generate_Dataset import generate_dataset, save_daily_dataset, generate_rebalanced_dataset

__author__ = 'bejar'



if __name__ == '__main__':
    z_factor = 0.25
    ldaysTr = ['20161101', '20161102', '20161103', '20161104', '20161107', '20161108', '20161109',
               '20161110', '20161111', '20161112', '20161113', '20161114', '20161115']
    # ldaysTs = ['20161107', '20161108', '20161109', '20161110', '20161111', '20161112', '20161113']
    ldaysTs = ['20161029', '20161030', '20161031', '20161101', '20161102', '20161103', '20161104',
               '20161105', '20161106', '20161107', '20161108', '20161109', '20161110', '20161111',
               '20161112', '20161113', '20161114', '20161115', '20161116', '20161117', '20161118',
               '20161119', '20161120', '20161121', '20161122']
    ncomp = 1000


    # X_train, y_train, X_test, y_test = generate_dataset(ldaysTr, ldaysTs, z_factor, PCA=True, ncomp=350)
    # np.save(data_path + 'train_data.npy', X_train)
    # np.save(data_path + 'train_labels.npy', np.array(y_train))
    # np.save(data_path + 'test_data.npy', X_test)
    # np.save(data_path + 'test_labels.npy', np.array(y_test))

    save_daily_dataset(ldaysTr, ldaysTs, z_factor=z_factor, ncomp=ncomp, method='two')

    # generate_rebalanced_dataset(ldaysTs, [(1,4), (2,3), (3, 15), (4,15), (5,15)], z_factor=z_factor, ncomp=ncomp)

