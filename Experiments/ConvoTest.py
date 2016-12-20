"""
.. module:: ConvoTest

ConvoTest
*************

:Description: ConvoTest

    

:Authors: bejar
    

:Version: 

:Created on: 28/11/2016 11:10 

"""

from keras import backend as K
K.set_image_dim_ordering('th')


from Models.SimpleModels import simple_model
from Util.ConvoTrain import transweights, train_model, load_dataset

__author__ = 'bejar'


if __name__ == '__main__':


    ldaysTr = ['20161102','20161103','20161104','20161105','20161106','20161107','20161108','20161109','20161110',
               '20161111', '20161112', '20161113', '20161114', '20161115', '20161116', '20161117', '20161118',
               '20161119', '20161120', '20161121', '20161122', '20161123']
    ldaysTs = ['20161124']
    z_factor = 0.25
    camera = None  #'Ronda' #Cameras[0]

    smodel = 3
    classweight = {0: 1.5, 1: 1, 2: 2.0, 3: 3.0, 4: 4.0}

    config = {'train': ldaysTr,
              'test': ldaysTs,
              'zfactor': 0.25,
              'model': smodel,
              'dpconvo': 0.2,
              'dpfull': 0.7,
              'convofields': [3, 3],
              'fulllayers': [64, 32],
              'classweight': transweights(classweight),
              'epochs': 100,
              'lrate': 0.005,
              'decay': 0.005/100,
              'batchsize': 100,
              'momentum': 0.9}

    train, test, test_labels, num_classes = load_dataset(ldaysTr, ldaysTs, z_factor, camera)

    config['input_shape'] = train[0][0].shape
    config['nexamples'] = train[0].shape[0]
    config['num_classes'] = num_classes

    model = simple_model(smodel, config)

    # Compile model

    train_model(model, config, train, test, test_labels, classweight)



