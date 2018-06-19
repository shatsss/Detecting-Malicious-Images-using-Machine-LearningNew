import cPickle
import mxnet as mx
import numpy as np
import cv2


def extractImagesAndLabels(path, file):
    f = open(path + file, 'rb')
    dict = cPickle.load(f)
    images = dict['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    imagearray = mx.nd.array(images)
    labelarray = mx.nd.array(labels)
    return imagearray, labelarray


def extractCategories(path, file):
    f = open(path + file, 'rb')
    dict = cPickle.load(f)
    return dict['label_names']


def saveCifarImage(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1, 2, 0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path + file + ".png", array)


def createImages():
    for j in range(1, 3):
        imgarray, lblarray = extractImagesAndLabels("cifar/", "data_batch_" + str(j))
        for i in range(0, 1000):
            saveCifarImage(imgarray[i], "./", "Images/image" + (str)(i + ((j - 1) * 1000)))
