import csv
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, Convolution2D

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

input_shape = (160, 320, 3)

def car_net():
    model_ = Sequential()
    model_.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model_.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model_.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2),  activation='relu'))
    model_.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
    model_.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
    model_.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model_.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model_.add(Flatten())
    model_.add(Dense(100))
    model_.add(Dropout(0.5))
    model_.add(Dense(50))
    model_.add(Dropout(0.5))
    model_.add(Dense(10))
    model_.add(Dropout(0.5))
    model_.add(Dense(1))
    return model_


def preprocess_input(data):
    data = data.astype(np.float32) / 255.0
    data -= 0.5
    return data


def generator(samples, measurements, marks, batch_size=32):
    num_samples = len(samples)
    while 1:
        samples, measurements, marks = shuffle(samples, measurements, marks)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            labels = measurements[offset:offset+batch_size]
            ms = marks[offset:offset + batch_size]
            images = []
            angles = []
            batch_count = len(batch_samples)
            for i in range(batch_count):
                image = cv2.imread(batch_samples[i])
                center_angle = float(labels[i])
                m = ms[i]
                if m == 1:
                    # flipping
                    image = cv2.flip(image, 1)
                    center_angle = center_angle * -1.0
                images.append(image)
                angles.append(center_angle)
            yield shuffle(np.array(images), np.array(angles))


def plot_loss_and_accuracy(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.legend(['acc', 'val_acc'])
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.show()