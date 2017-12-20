import numpy as np
import cv2


# Data augmentation functions

def toss(prob):
    if prob == 1:
        return True
    elif prob == 0:
        return False
    else:
        return np.random.random() < prob


def augTranslate(image, xMax, xProb, yMax, yProb):
    rows, cols = image.shape[:2]
    xMax, yMax = int(xMax*cols), int(yMax*rows)
    xTrans = np.random.randint(-xMax,xMax) if toss(xProb) else 0
    yTrans = np.random.randint(-yMax,yMax) if toss(yProb) else 0

    M = np.float32([[1, 0, xTrans],[0, 1, yTrans]])
    image = cv2.warpAffine(image,M,(cols,rows))

    return image


def augFlip(image, command, prob):
    if toss(prob):
        image = cv2.flip(image, 1)
        command = -command

    return image, command


def _genRandomShadowMask(shape):
    rows, cols, _ = shape

    # pick random points from horizontal edges
    [x1, x2] = np.random.choice(cols, 2, replace=False)
    m = rows/(x2 - x1)
    c = - m * x1
    # construct a mask
    x = np.mgrid[0:rows, 0:cols][1]
    y = np.mgrid[0:rows, 0:cols][0]
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[(m*x + c - y <= 0)] = 1.0

    return mask

def _brighten(vdata, factor):
    vdata = cv2.multiply(vdata, np.array([factor]))
    vdata[vdata > 255] = 255
    return vdata


def augBright(image, brMax, brProb, shVal=0, shProb=0):

    img2d = (image.ndim == 2)
    toBr, toSh = toss(brProb), toss(shProb)

    if toBr or toSh:
        if img2d:
            v = image
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)

        if toBr:
            brVal = 1 + np.random.random()*2*brMax - brMax
            v = _brighten(v, brVal)

        if toSh:
            mask = _genRandomShadowMask(image.shape)
            v = mask*_brighten(v, 1-shVal) + (1-mask)*v

        if img2d:
            image = v
        else:
            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return image

    else:
        return image


def augDrop(command, threshold, prob):
    """
    If command < threshold, drop the data with a probability prob
    """
    return abs(command) < threshold and toss(prob)



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    img = cv2.imread('center_2016_12_01_13_31_13_686.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Test flip
    sqNum = 10
    fig = plt.figure()
    fig.canvas.set_window_title('Flip augmentation')
    for i in range(sqNum*sqNum):
        ax = plt.subplot(sqNum, sqNum, i+1)
        resI, resC = augFlip(img, 5, 0.75)
        resI = np.array(resI)
        plt.imshow(resI)
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # Test Bright
    fig = plt.figure()
    fig.canvas.set_window_title('Brightness augmentation')
    sqNum = 10
    for i in range(sqNum*sqNum):
        ax = plt.subplot(sqNum, sqNum, i+1)
        resI = augBright(img, brMax=0.5, brProb=0.95)
        resI = np.array(resI)
        plt.imshow(resI)
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # Test translations
    fig = plt.figure()
    fig.canvas.set_window_title('Translation augmentation')
    sqNum = 10
    for i in range(sqNum*sqNum):
        ax = plt.subplot(sqNum, sqNum, i+1)
        resI = augTranslate(img, 0.25, 0, 0.25, 1)
        resI = np.array(resI)
        plt.imshow(resI)
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # Test shadows
    fig = plt.figure()
    fig.canvas.set_window_title('Brightness and shadow augmentation')
    sqNum = 10
    for i in range(sqNum*sqNum):
        ax = plt.subplot(sqNum, sqNum, i+1)
        resI = augBright(img, brMax=0.5, brProb=1, shVal=0.5, shProb=1)
        resI = np.array(resI)
        plt.imshow(resI)
        plt.xticks([])
        plt.yticks([])
    plt.show()
