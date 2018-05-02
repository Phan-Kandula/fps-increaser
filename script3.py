import numpy as np
import time
import os
import tensorflow as tf
import keras

'''
These scripts are supposed to help convert 30 fps videos to 60 fps videos by guessing the image in between using a deep
net. The data set consists of frames of video converted into a numpy array. 
'''

'''
TODO
make X be an array of the flattened images and the difference
'''
def create_training_set(arr):
    i = 1
    xr = np.zeros((len(arr) - 2, len(arr[0]),len(arr[0][0]),len(arr[0][0][0])))
    xl = np.zeros((len(arr) - 2, len(arr[0]),len(arr[0][0]),len(arr[0][0][0])))
    y1 = np.zeros((len(arr) - 2, len(arr[0]),len(arr[0][0]),len(arr[0][0][0])))
    while i <= (len(arr) - 2):
        np.put(xl, i - 1, arr[i - 1])
        np.put(xr, i - 1, arr[i + 1])
        np.put(y1, i - 1, arr[i])
        '''
        xl[i-1] = arr[i-1]
        xr[i-1] = arr[i+1]
        y1[i-1] = arr[i]
        '''
        i = i + 1
    if len(xr > 0):
        print("xr loaded")
    else:
        print("xr not loaded")


    n_pixels = len(arr[0]) * len(arr[0][0]) * len(arr[0][0][0])
    X = np.zeros((len(arr)-2,3,n_pixels))
    for i in range(len(xr)):
        diff = np.array([xr[i]-xl[i]]).flatten()
        np.put(X,i,np.array([[xr[i].reshape([1,691200])],[diff], [xl[i].reshape([1,691200])]]))
    return X,y1

def next_batch(arr, initial, size):
    if (initial + size )>= len(arr):
        return arr[initial:]
    return arr[initial: initial + size]

def diff_images(actual, prediction, name):
    return tf.square(tf.subtract(actual, prediction), name= name)

def weight_variable(shape, name):
    var = tf.truncated_normal(shape, stddev=0.1)
    return tf.variable(var, name=name)

def bias_variable(shape, name):
    var = tf.truncated_normal(shape, stddev=0.1)
    return tf.variable(var, name=name)

def layer(X, W, b):
    return tf.matmul(X,W) + b


if __name__ == '__main__':
    start_time = time.time()
    data = np.load(os.getcwd() + '/train_set1.npy')
    if len(data) > 0:
        print("Data loaded")
    else:
        print(os.getcwd() + '/train_short_set1.npy hasn\'t loaded')
    print(str(len(data)) + ' ' + str(len(data[0])) + ' ' +
            str(len(data[0][0])) + ' ' + str(len(data[0][0][0])))
    print("--- %s seconds to load data ---" % (time.time() - start_time))

    n_pixels = len(data[0]) * len(data[0][0]) * len(data[0][0][0])

    X = tf.placeholder(tf.float32, shape=([None, len(data[0]), len(data[0][0]), len(data[0][0][0])]))

    arr = create_training_set(next_batch(data,0,50))

