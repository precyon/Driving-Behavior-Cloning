import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
from  sklearn import model_selection
from keras import models, optimizers, backend
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Cropping2D



settings = {
        'path': '.\data',
        'id': '0',
        'shape': (160, 320, 3),
        'size': 8036
        }

batchSize = 128

def readDataFile():
    dataPath = os.path.join(settings['path'], settings['id'])
    drvData = pd.io.parsers.read_csv(os.path.join(dataPath, 'driving_log.csv'))
    return drvData

def inputGenerator(dfData):
    lenData = dfData.shape[0]
    while True:
        for bStart in range(0, lenData, batchSize):
            imgData = np.empty([0, 160, 320, 3], dtype=np.float32)
            strData = np.empty([0], dtype=np.float32)

            for i in range(bStart, min(bStart + batchSize, lenData)):
                image = img.imread(os.path.join(settings['path'], settings['id'], dfData['center'].values[i]))
                imgData = np.append(imgData, [image], axis=0)
                strData = np.append(strData, dfData['steering'].values[i])

            yield imgData, strData

def preProcessor(img):
    return (img/255.0) - 0.5

def mLinear():
    model = models.Sequential()
    model.add(Lambda(preProcessor, input_shape = settings['shape']))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model

def mLenet():
    model = models.Sequential()
    model.add(Lambda(preProcessor, input_shape = settings['shape']))
    model.add(Cropping2D(cropping = ((70, 25), (0, 0))))
    model.add(Conv2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model

if __name__ == '__main__':

    drvData = readDataFile()
    #imgData, strData = readFiles(drvData)

    dfValid, dfTrain = model_selection.train_test_split(drvData, test_size=-.2)

    model = mLenet()

    trHistory = model.fit_generator(
            inputGenerator(dfTrain),
            samples_per_epoch = dfTrain.shape[0],
            nb_epoch = 5,
            validation_data = inputGenerator(dfValid),
            nb_val_samples = dfValid.shape[0]
            )

    model.save('model.h5')

    # Plot the history
    plt.plot(trHistory.history['loss'])
    plt.plot(trHistory.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
