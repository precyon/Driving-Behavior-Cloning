import math
import numpy as np
from keras import models, optimizers, backend
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Cropping2D


class kModel(object):

    def __init__(self):
        self.model = None

    def compile(self, batchSize, learning_rate = 1e-4, epochs = 5, optimizer = 'adam'):
        if self.model == None:
            raise Exception("Model not defined")
        self.summary = None
        self.history = None
        self.batchSize = batchSize
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.model.compile(loss='mse', optimizer=self.optimizer)

    def updateSummary(self, data):
        self.summary = np.append(self.summary, data)

    def train(self, trainingGen, trainingData, validationGen, validationData, augment=True):
        if self.model == None:
            raise Exception("Model not defined")

        trainingDataSize = trainingData.shape[0]
        validationDataSize = validationData.shape[0]

        samplesPerEpoch = math.ceil(trainingDataSize/self.batchSize)*self.batchSize

        self.summary = np.empty([0])
        self.history = self.model.fit_generator(
                trainingGen(trainingData, augment=augment, callback=self.updateSummary),
                samples_per_epoch = samplesPerEpoch,
                nb_epoch = self.epochs,
                validation_data = validationGen(validationData),
                nb_val_samples = validationDataSize
                )

    def save(self):
        if self.model == None:
            raise Exception("Model not defined")
        self.model.save('model.h5')
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

    def __init__(self, input_shape, preprocessor):
        # Build the model
        model = models.Sequential()
        model.add(Lambda(preprocessor, input_shape = input_shape))
        model.add(Cropping2D(cropping = ((70, 25), (0, 0))))
        model.add(Conv2D(6, 5, 5, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(6, 5, 5, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Dense(84))
        model.add(Dense(1))

        self.model = model
