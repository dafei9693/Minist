import matplotlib.pyplot as plt
import numpy as np
import struct
from PIL import Image


def loadImageSet(filename):

    # Read file
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    # Get the first four integer, return a tuplie
    head = struct.unpack_from('>IIII', buffers, 0)

    # Reach the place of data begin
    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    # print(imgNum)
    width = head[2]
    # print(width)
    height = head[3]
    # print(height)

    # data -> 60000*28*28
    bits = imgNum * width * height
    # fmt format：'>47040000B'
    bitsString = '>' + str(bits) + 'B'

    # Get data，return a tuple
    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    # Reshape to array of [60000,784]
    imgs = np.reshape(imgs, [imgNum, width * height])

    return imgs


def loadLabelSet(filename):

    # Read file
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    # Get the first two integer of label file
    head = struct.unpack_from('>II', buffers, 0)

    # Reach the place of the label data begin
    labelNum = head[1]
    offset = struct.calcsize('>II')

    # fmt format：'>60000B'
    numString = '>' + str(labelNum) + "B"
    # Get label
    labels = struct.unpack_from(numString, buffers, offset)

    binfile.close()
    # Reshape to an array
    labels = np.reshape(labels, [labelNum])

    return labels
