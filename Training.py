#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
import numpy as np
import pandas as pd
import os
import sys
import sklearn
import datetime
import random
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D  
from tensorflow.python.keras.backend import batch_normalization
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from packaging import version

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# In[3]:


#identify GPU
device_name = tf.test.gpu_device_name()
if not tf.test.is_gpu_available():
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# In[4]:


print("TensorFlow version: ", tf.__version__)


# In[5]:


#train data
X_train = np.load(os.path.join("Data", "X_train.npy"))
y_train = np.load(os.path.join("Data", "train_labels_multi.npy"))
y_train_bi = np.load(os.path.join("Data", "y_train.npy"))

#test data
X_test = np.load(os.path.join("Data", "X_test.npy"))
y_test = np.load(os.path.join("Data", "y_test_labels_multi.npy"))
y_test_bi = np.load(os.path.join("Data", "y_test.npy"))

#validation data
X_val = np.load(os.path.join("Data", "X_val.npy"))
y_val = np.load(os.path.join("Data", "y_val_labels_multi.npy"))
y_val_bi = np.load(os.path.join("Data", "y_val.npy"))


# In[6]:


#train data
print("X_train data:", X_train.shape)
print("y_train data:", y_train.shape)


# In[7]:


print(y_train.shape)


# In[8]:


#validation data
print("X_validation data:", X_val.shape)
print("y_validation data:", y_val.shape)


# In[9]:


#test data
print("X_test data:", X_test.shape)
print("y_test data:", y_test.shape)


# In[10]:


y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

#binary
y_train_bi = to_categorical(y_train_bi)
y_val_bi = to_categorical(y_val_bi)
y_test_bi = to_categorical(y_test_bi)


# In[11]:


# scale pixels
X_train = X_train/255.0
X_val = X_val/255.0
X_test = X_test/255.0


# In[12]:


classes = 4
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=X_train.shape[1:]))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(classes, activation='softmax'))
    # compile model
    opt = Adam(lr=0.001, beta_1=0.9, beta_2 = 0.999)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[13]:


model = define_model()


# In[14]:


model.summary()


# In[ ]:


# fit model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))


# In[ ]:


_, acc = model.evaluate(X_test, y_test, verbose=0)
print('> %.3f' % (acc * 100.0))

