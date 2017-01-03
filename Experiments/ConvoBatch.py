"""
.. module:: ConvoBatch

ConvoBatch
*************

:Description: ConvoBatch

    

:Authors: bejar
    

:Version: 

:Created on: 23/12/2016 15:05 

"""

__author__ = 'bejar'


from keras import backend as K
K.set_image_dim_ordering('th')


from Models.SimpleModels import simple_model
from Util.ConvoTrain import transweights, train_model_batch
from Util.Generate_Dataset import list_days_generator
from Util.ConvoTrain import load_dataset

__author__ = 'bejar'


if __name__ == '__main__':


    ldaysTr = list_days_generator(2016, 11, 1, 30) 
    ldaysTs = list_days_generator(2016, 12, 1, 2)
    z_factor = 0.25
    camera = None  #'Ronda' #Cameras[0]

    smodel = 4
    classweight = {0: 1.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}

    config = {'train': ldaysTr,
              'test': ldaysTs,
              'reblanced': False,
              'zfactor': 0.25,
              'model': smodel,
              'dpconvo': 0.2,
              'dpfull': 0.4,
              'convofields': [3, 3],
              'fulllayers': [64, 32],
              'convolayers': [128, 64, 64],
              'classweight': transweights(classweight),
              'epochs': 200,
              'lrate': 0.005,
              'decay': 0.005/200,
              'batchsize': 200,
              'momentum': 0.9}

    _, test, test_labels, num_classes = load_dataset(ldaysTr, ldaysTs, z_factor, gen=False, only_test=True)

    config['input_shape'] = test[0][0].shape
    config['num_classes'] = num_classes

    model = simple_model(smodel, config)

    train_model_batch(model, config, test, test_labels)


