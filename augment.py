import numpy as np
import cv2


# Data augmentation functions

def toss(prob):
    return True if prob == 1 else np.random.random() < prob


def augTranslate(image, xMax, xProb, yMax, yProb):
    rows, cols, _ = image.shape
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


def augBright(image, brMax, prob, constant = False):
    if toss(prob):
        brVal = brMax if constant else 1 + np.random.random()*2*brMax - brMax
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.multiply(v, np.array([brVal]))
        v[v > 255] = 255
        hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image

def augShadow(image, dimVal, prob):
    if toss(prob):
        rows, cols, chan = image.shape
        # choose a shadow intensity
        dimImg = augBright(image, dimVal, prob = 1, constant = True)
        # pick random points from horizontal edges
        [x1, x2] = np.random.choice(cols, 2, replace=False)
        m = rows/(x2 - x1)
        c = - m * x1
        # construct a mask
        x = np.mgrid[0:rows, 0:cols][1]
        y = np.mgrid[0:rows, 0:cols][0]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        mask[(m*x + c - y <= 0)] = 1.0
        # Apply the mask
        image = mask[:,:,None]*dimImg + (1 - mask[:,:,None])*image

    return image




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
        resI = augBright(img, 0.5, 0.95)
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
    fig.canvas.set_window_title('Shadow augmentation')
    sqNum = 10
    for i in range(sqNum*sqNum):
        ax = plt.subplot(sqNum, sqNum, i+1)
        resI = augShadow(img, 0.5, 0.95)
        resI = np.array(resI)
        plt.imshow(resI)
        plt.xticks([])
        plt.yticks([])
    plt.show()
