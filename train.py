import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from sklearn import model_selection
from keras import models, optimizers, backend
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Cropping2D

import modelzoo as zoo
from augment import augFlip, augBright, augDrop


settings = {
        'path': '.\data',
        'id': 'Q3RQQS',
        'readShape': (160, 320, 3),
        'preShape': (64,64,3)
        }

cameras = ['left', 'center', 'right']
steering = [0.25, 0, -0.25]

batchSize = 128

def cleanDataFile(dfData):
    lenData = dfData.shape[0]
    dfData.columns=['center','left','right','steering','throttle','brake','speed']
    for i in range(lenData):
        for j in range(3):
            dfData[dfData.columns[j]].values[i] = os.path.basename(dfData[dfData.columns[j]].values[i])
    return dfData

def readDataFile():
    dataPath = os.path.join(settings['path'], settings['id'])
    fPath    = os.path.join(dataPath, 'driving_log_processed.csv')
    if not os.path.isfile(fPath):
        drvData = pd.io.parsers.read_csv(os.path.join(dataPath, 'driving_log.csv'))
        drvData = cleanDataFile(drvData)
        drvData.to_csv(os.path.join(dataPath, 'driving_log_processed.csv'), index=False)
    else:
        drvData = pd.io.parsers.read_csv(os.path.join(dataPath, 'driving_log_processed.csv'))

    return drvData

def inputGenerator(dfData, augment=True, callback=None):
    lenData = dfData.shape[0]
    imgData = np.zeros([batchSize, *settings['readShape']], dtype=np.float32)
    strData = np.zeros([batchSize])

    while True:
        # supply the next batch of an appropriate size
        for i in range(batchSize):

            # Randomly pickup a line from the dataframe till the drop condition is not met
            while True:
                line = np.random.randint(lenData)
                command = dfData['steering'].values[line]
                if not (augment and augDrop(command, threshold=0.1, prob = 0.95)):
                    break

            # Randomly choose a camera, correct the steering command and load the image
            camera = np.random.randint(len(cameras)) if augment else 1
            command += steering[camera]

            imgPath = os.path.join(settings['path'], settings['id'], 'IMG', dfData[cameras[camera]].values[line].strip())
            image = cv2.imread(imgPath)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if augment:

                 # Add random brightness changes and shadows
                 #image = augBright(image, 0.25, 0.95, shVal=0.5, shProb=0.5)

                 # Randomly translate the image horizontally

                 # Flip the image horizontally
                 image, command = augFlip(image, command, prob = 0.5)

            # Convert to np.array
            image = np.array(image)
            imgData[i, :, :, :] = image
            strData[i] = command

        # Remember the steering command for statistics
        if callback is not None:
            callback(strData)

        yield imgData, strData


def validationGenerator(dfData):
    lenData = dfData.shape[0]
    imgData = np.zeros([batchSize, *settings['readShape']], dtype=np.float32)
    strData = np.zeros([batchSize])
    while True:
        for bStart in range(0, lenData, batchSize):

            for j in range(batchSize):
                # Randomly choose a camera
                camera = np.random.randint(len(cameras))

                # Load the image and read the command
                i = bStart + j
                imgPath = os.path.join(settings['path'], settings['id'], 'IMG', dfData[cameras[camera]].values[i].strip())
                image = cv2.imread(imgPath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                command = dfData['steering'].values[i] + steering[camera]

                image = np.array(image)
                imgData[j,:,:,:] = image
                strData[j] = command

            yield imgData, strData


def preProcessor(img):
    # Crop
    # img = img[60:-25,:,:]
    # Resize
    # img = cv2.resize(img, settings['preShape'][0:2], interpolation = cv2.INTER_AREA)
    # Normalize and return
    return (img/255.0) - 0.5

if __name__ == '__main__':

    #img = cv2.imread('center_2016_12_01_13_31_13_686.jpg')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(preProcessor(img))
    #plt.show()


    drvData = readDataFile()
    #drvData = drvData.head(1024)

    dfTrain, dfValid = model_selection.train_test_split(drvData,
            test_size = int(np.floor(drvData.shape[0]*0.15/batchSize)*batchSize)
            )


    zmodel = zoo.mLeNet(input_shape=settings['readShape'], preprocessor=preProcessor)
    zmodel.compile(batchSize, epochs = 5)
    zmodel.train(inputGenerator, dfTrain, validationGenerator, dfValid, augment=True)
    zmodel.save()

    # Plot the logged summary
    fig = plt.figure
    plt.hist(zmodel.summary, bins=100)
    plt.show()
#    # Plot the history
#    plt.plot(trHistory.history['loss'])
#    plt.plot(trHistory.history['val_loss'])
#    plt.title('model mean squared error loss')
#    plt.ylabel('mean squared error loss')
#    plt.xlabel('epoch')
#    plt.legend(['training set', 'validation set'], loc='upper right')
#    plt.show()

