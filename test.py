import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import tensorflow as tf
from tensorflow import keras as kr

path = "datasets/data/"

labels_f = pd.read_csv('datasets/labels.csv', delimiter=';')

categories = np.array(labels_f.iloc[:, 0])

"""Since images have different shapes, let's resize pictures to height = 32 and width = 55.

<h1>3. Preprocess data and label inputs</h1>
"""

# initialize the data and labels
data = []
labels = []
imagePaths = []
HEIGHT = 128
WIDTH = 128

N_CHANNELS = 3

# grab the image paths and randomly shuffle them
for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k])

import random
random.shuffle(imagePaths)
print(imagePaths[:10])

# loop over the input images
for imagePath in imagePaths:
    # load the image, resize the image to be HEIGHT * WIDTH pixels (ignoring
    # aspect ratio) and store the image in the data list
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath[1]
    labels.append(label)



# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Let's check everything is ok
plt.subplots(3,4)
for i in range(12):
    plt.subplot(3,4, i+1)
    plt.imshow(data[i])
    plt.axis('off')
    plt.title(categories[labels[i]])
plt.show()

"""<h1>4. Split dataset into train and test set</h1>"""

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
# Preprocess class labels
trainY = kr.utils.to_categorical(trainY, len(categories))

"""<h1>5. Define model architecture</h1>"""

EPOCHS = 20
INIT_LR = 1e-3
BS = 4
#--------------------------------------------
class_names = categories
#--------------------------------------------

from keras import layers
from keras import models


print("[INFO] compiling model...")
mobileNet = kr.applications.MobileNet(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
for layer in mobileNet.layers:
    layer.trainable = False

model = Sequential()
model.add(mobileNet)
model.add(kr.layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(len(class_names), activation='softmax'))

opt = tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

#Training
model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, verbose=1)

model.save_weights("model.h5")

"""<h1>7. Evaluate model on test data</h1>"""

from numpy import argmax
from sklearn.metrics import confusion_matrix, accuracy_score

pred = model.predict(testX)
predictions = argmax(pred, axis=1) # return to label

cm = confusion_matrix(testY, predictions)

accuracy = accuracy_score(testY, predictions)
print("Accuracy : %.2f%%" % (accuracy*100.0))

img_path="test4.jpg"


img = kr.preprocessing.image.load_img(img_path, target_size=(128, 128))
img_array = kr.preprocessing.image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = kr.applications.mobilenet.preprocess_input(img_batch)

pred = model.predict(img_preprocessed)
Res = argmax(pred, axis=1) # return to label
print(pred)



Result_Text = "{0}({1})".format(categories[Res[0]],round(pred[0][Res[0]]*100,2))

plt.text(10, 10, Result_Text, color="blue",fontsize="large",bbox=dict(fill=False, edgecolor='red', linewidth=1))
plt.imshow(img)
plt.show()

print(categories[Res[0]],pred[0][Res[0]]*100)
