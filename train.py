import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
from  sklearn import model_selection
from keras import models, optimizers, backend
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Cropping2D

import modelzoo as zoo


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

if __name__ == '__main__':

    drvData = readDataFile()
    #imgData, strData = readFiles(drvData)

    dfValid, dfTrain = model_selection.train_test_split(drvData, test_size=-.2)

    #model = mLenet()

    zmodel = zoo.mLeNet(input_shape=settings['shape'], preprocessor = preProcessor)
    zmodel.compile()
    zmodel.train(inputGenerator, dfTrain, dfValid)
    zmodel.save()

#    # Plot the history
#    plt.plot(trHistory.history['loss'])
#    plt.plot(trHistory.history['val_loss'])
#    plt.title('model mean squared error loss')
#    plt.ylabel('mean squared error loss')
#    plt.xlabel('epoch')
#    plt.legend(['training set', 'validation set'], loc='upper right')
#    plt.show()
