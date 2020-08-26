import tensorflow as tf
print("TensorFlow is Running...")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils1 import *

#Training Image path
path = "natural_images\Training"
classes = os.listdir(path)
print(classes)

from PIL import Image
import numpy as np

# Loading Images from directory and storing in list
image_count = list()
images__ = list()
for images in classes:
    images_path = path+"/"+images
    images_ = os.listdir(images_path)
    image_count.append(images_)
    for open_img in images_:
        open_image = images_path+"/"+open_img
        print("Loading image... ",open_image)
        image = Image.open(open_image)
#         print(image)
        resized = image.resize((120,120))
        image_array = np.array(resized)
        images__.append(np.array(image_array))

feature = np.array(images__)
print(feature.shape)

# Performing one hot encoding
from keras.utils import to_categorical


classes_to_int = list()
for i in range(len(classes)):
    classes_to_int.append(i)

# Changing classes in one hot encoding
one_hot_encoding = to_categorical(classes_to_int)
print(one_hot_encoding)

label = list()
for i in range(len(classes)):
    for j in range(len(image_count[i])):
        label.append(one_hot_encoding[i])

labels = np.array(label)
print(labels.shape)


# index = 1253

# print(classes[np.argmax(labels[index])])
# plt.imshow(images__[index])
# plt.show()

from random import sample
def train_test_creation(x, data, toPred): 
    indices = sample(range(data.shape[0]),int(x * data.shape[0])) 
    indices = np.sort(indices, axis=None) 
  
    index = np.arange(data.shape[0]) 
    reverse_index = np.delete(index, indices,0)
  
    train_toUse = data[indices]
    train_toPred = toPred[indices]
    test_toUse = data[reverse_index]
    test_toPred = toPred[reverse_index]

    return train_toUse, train_toPred, test_toUse, test_toPred

xtrain, ytrain, xtest, ytest = train_test_creation(0.7, feature, labels)
print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape )

XTRAIN = xtrain/255.
XTEST = xtest/255.
YTRAIN = ytrain
YTEST = ytest

print ("X_train shape: " + str(XTRAIN.shape))
print ("Y_train shape: " + str(YTRAIN.shape))
print ("X_test shape: " + str(XTEST.shape))
print ("Y_test shape: " + str(YTEST.shape))
