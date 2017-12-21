import cv2
import argparse
import numpy as np
from keras.models import load_model
from moviepy.editor import VideoFileClip
from keras import models
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout, ELU

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from train import settings, preProcessor
import modelzoo as zoo

hmodel = None
dw = None

def activations(img):
    preimg = np.reshape(preProcessor(img), settings['preShape'])
    hout = hmodel.predict(preimg[None,...], batch_size=1)
    act = np.multiply(hout.T, dw[0]) + dw[1]
    act = np.reshape(act, (14,30,3))

    selectChan = np.absolute(act[:,:,1])
    actImg = (selectChan/np.max(selectChan))

    overlay = cv2.resize(actImg, (320, 75), interpolation=cv2.INTER_LINEAR)
    overlay = (overlay*255).astype(np.uint8)
    overimg = np.zeros(img.shape[:2], dtype=np.uint8)
    overimg[60:-25,:] = overlay
    overimg = cv2.applyColorMap(overimg, cv2.COLORMAP_JET)

    result = cv2.addWeighted(overimg, 0.5, img, 0.5, 0)

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )

    parser.add_argument(
        'video',
        type=str,
        nargs='?',
        default='',
        help='Path to the input video'
    )
    args = parser.parse_args()

    fModel = load_model(args.model)
    fModel.summary()
    denseWeights = fModel.layers[5].get_weights()

    hmodel = models.Sequential()
    hmodel.add(Lambda(lambda x: (x/255 - 0.5)*2, input_shape = settings['preShape'],
            name='Normalize'))
    hmodel.add(Conv2D(3, 5, 5, activation='elu', name='Conv',
        weights=fModel.layers[1].get_weights()))
    hmodel.add(MaxPooling2D( name='MaxPool'))
    hmodel.add(Flatten( name='Flatten'))


    dw = denseWeights
    videoIn = VideoFileClip(args.video)
    videoOut = videoIn.fl_image(activations)
    videoOut.write_videofile('out.mp4', audio=False)
    print('Ouput saved as out.mp4')

