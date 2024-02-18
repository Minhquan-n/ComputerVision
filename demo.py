import numpy as np
import pandas as pd
import random
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Activation, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from tensorflow import keras

# set up gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=7168)])
    logical_gpus = tf.config.list_logical_devices('GPU')
  except RuntimeError as e:
    print(e)

# width and height to resize image 254x254
WIDTH = 128
HEIGHT = 128

EPOCHS = 20
INIT_LR = 1e-3
BS = 8

def create_data_path_list(dataDirPath, labels):
    imgPaths = []
    for number, name in enumerate(labels):
        categoryPath = dataDirPath + name
        for path in os.listdir(categoryPath):
            imgPaths.append([categoryPath + '/' + path, number])

    random.shuffle(imgPaths)
    return imgPaths

def create_training_data(dataDirPath, labels):
    X = []
    Y = []
    imgPaths = create_data_path_list(dataDirPath, labels)

    for path in imgPaths:
        img = cv2.imread(path[0], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (HEIGHT, WIDTH))
        if img is not None:
            X.append(img)
            Y.append(path[1])
    
    return np.array(X, dtype='float32') / 255.0, np.array(Y)

dirPath = 'datasets/'
dataDirPath = dirPath + 'data/'

categories = pd.read_csv(dirPath + 'labels.csv', delimiter=';')

labels = np.array(categories.iloc[:, 0])

X, Y = create_training_data(dataDirPath, labels)

# create train and test
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
trainY = keras.utils.to_categorical(trainY, len(labels))

# create train and validation
trainX, validateX, trainY, validateY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

mobileNet = keras.applications.MobileNet(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
for layer in mobileNet.layers:
    layer.trainable = False

# create model to extract features
model = Sequential()
model.add(mobileNet)
#model.add(layers.AveragePooling2D((8, 8), padding='valid', name='avg_pool'))
model.add(keras.layers.GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(len(labels), activation='softmax'))

opt = tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

#Training
model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, verbose=1)

model.save_weights("model.h5")

pred = model.predict(testX)

acc = accuracy_score(testY, pred)

print(acc * 100)


