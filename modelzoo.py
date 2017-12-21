import math
import numpy as np
import tensorflow as tf
from keras import models, optimizers, backend
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout, ELU
from keras.backend import tf as ktf

def atan(x):
    return tf.atan(x)

class kModel(object):

    def __init__(self):
        self.model = None

    def compile(self, batchSize, learning_rate = 1e-4, epochs = 5, optimizer = 'adam'):
        if self.model == None:
            raise Exception("Model not defined")
        self.log = None
        self.history = None
        self.batchSize = batchSize
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.model.compile(loss='mse', optimizer=self.optimizer)

    def updateSummary(self, data):
        self.log = np.append(self.log, data)

    def train(self, trainingGen, trainingData, validationGen, validationData, augment=True):
        if self.model == None:
            raise Exception("Model not defined")

        trainingDataSize = trainingData.shape[0]
        validationDataSize = validationData.shape[0]

        samplesPerEpoch = math.ceil(trainingDataSize/self.batchSize)*self.batchSize

        self.log = np.empty([0])
        self.history = self.model.fit_generator(
                trainingGen(trainingData, augment=augment, callback=self.updateSummary),
                samples_per_epoch = samplesPerEpoch,
                nb_epoch = self.epochs,
                validation_data = validationGen(validationData),
                nb_val_samples = validationDataSize
                )

    def summary(self):
        self.model.summary()

    def save(self):
        if self.model == None:
            raise Exception("Model not defined")
        self.model.save('.\models\model.h5')
        print('Model saved as model.h5')


class mLinear(kModel):

    def __init__(self, input_shape, preprocessor):
        # Build the model
        model = models.Sequential()

        model.add(Lambda(preprocessor, input_shape = input_shape))

        model.add(Flatten())
        model.add(Dense(1))

        self.model = model


class mLeNet(kModel):


    def __init__(self, input_shape):

        model = models.Sequential()

        model.add(Lambda(lambda x: (x/255 - 0.5)*2, input_shape = input_shape,
                name='Normalize'))

        model.add(Conv2D(6, 5, 5, activation='relu', name='Conv'))
        model.add(MaxPooling2D( name='MaxPool'))
        model.add(Flatten( name='Flatten'))
        model.add(Dense(120, activation='relu', name='Dense1'))
        model.add(Dropout(0.5, name='Dropout1'))
        model.add(Dense(84, activation='relu', name='Dense2'))
        model.add(Dropout(0.5, name='Dropout2'))
        model.add(Dense(1, name='Output'))

        self.model = model

class mSmall(kModel):

    def __init__(self, input_shape):

        model = models.Sequential()

        model.add(Lambda(lambda x: (x/255 - 0.5)*2, input_shape = input_shape,
                name='Normalize'))

        model.add(Conv2D(3, 5, 5, activation='elu', name='Conv'))
        model.add(MaxPooling2D( name='MaxPool'))
        model.add(Flatten( name='Flatten'))
        model.add(Dropout(0.5, name='Dropout'))
        model.add(Dense(1, name ='Output'))

        self.model = model


class mComma(kModel):

    def __init__(self, input_shape):
        # Model from https://github.com/commaai/research/blob/master/train_steering_model.py
        model = models.Sequential()

        model.add(Lambda(lambda x: (x/255 - 0.5)*2, input_shape = input_shape))

        model.add(Conv2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='elu'))
        model.add(Conv2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu'))
        model.add(Conv2D(64, 5, 5, subsample=(2, 2), border_mode="same", activation='elu'))
        model.add(Flatten())
        model.add(Dropout(.3))
        model.add(Dense(512, activation='elu'))
        model.add(Dropout(.5))
        model.add(Dense(1))

        self.model = model

class mNvidia(kModel):

    def __init__(self, input_shape):
        # Model from https://github.com/0bserver07/Nvidia-Autopilot-Keras/blob/master/model.py
        model = models.Sequential()

        model.add(Lambda(lambda x: (x/255 - 0.5)*2, input_shape = input_shape))

        model.add(Conv2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
        model.add(Conv2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
        model.add(Conv2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
        model.add(Conv2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
        model.add(Conv2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
        model.add(Flatten())
        model.add(Dense(1164, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='tanh'))

        self.model = model


