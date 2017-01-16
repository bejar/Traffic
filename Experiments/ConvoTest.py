"""
.. module:: ConvoTest

ConvoTest
*************

:Description: ConvoTest

    

:Authors: bejar
    

:Version: 

:Created on: 28/11/2016 11:10 

"""


########################### Outdated, use ConvoBatch.py

from keras import backend as K
K.set_image_dim_ordering('th')


from Models.SimpleModels import simple_model
from Util.ConvoTrain import transweights, train_model, load_dataset
from Util.Generate_Dataset import list_days_generator
from Util.DataGenerators import simpleDataGenerator
__author__ = 'bejar'


if __name__ == '__main__':


    ldaysTr = list_days_generator(2016, 11, 1, 23)
    ldaysTs = list_days_generator(2016, 11, 24, 24)
    z_factor = 0.25
    camera = None  #'Ronda' #Cameras[0]

    smodel = 3
    classweight = {0: 1.5, 1: 1, 2: 2.0, 3: 3.0, 4: 4.0}

    config = {'train': ldaysTr,
              'test': ldaysTs,
              'zfactor': 0.25,
              'model': smodel,
              'dpconvo': 0.4,
              'regfull': ['drop', 0.6],
              'convofields': [3, 3],
              'fulllayers': [64, 32],
              'classweight': transweights(classweight),
              'epochs': 100,
              'lrate': 0.005,
              'decay': 0.005/100,
              'batchsize': 100,
              'momentum': 0.9}

    generator = False

    if not generator:
        train, test, test_labels, num_classes = load_dataset(ldaysTr, ldaysTs, z_factor, gen=False)
        config['input_shape'] = train[0][0].shape
        config['nexamples'] = train[0].shape[0]
        config['num_classes'] = num_classes
        datagen = None
        samples_epoch = train[0].shape[0]
    else:
        train, test, test_labels, num_classes = load_dataset(ldaysTr, ldaysTs, z_factor, gen=False, only_test=True)
        datagen = simpleDataGenerator(ldaysTr, z_factor, num_classes, config['batchsize'], groups=4)
        samples_epoch = 34000
        config['input_shape'] = test[0][0].shape
        config['num_classes'] = num_classes
        config['nexamples'] = samples_epoch

    model = simple_model(smodel, config)

    # Compile model

    train_model(model, config, train, test, test_labels, generator=datagen, samples_epoch=samples_epoch)



