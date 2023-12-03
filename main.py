import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.losses import BinaryCrossentropy
import os
import cv2
import imghdr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_logical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#Load Data
tf.data.Dataset

#Builds data pipeline
data = tf.keras.utils.image_dataset_from_directory('PetImages/Train')
#Allows us to access data pipeline
data_iterator = data.as_numpy_iterator()
#Accessing data pipeline itself
batch = data_iterator.next()

#Scale Data
data_scaled = data.map(lambda x,y: (x/255, y))
scaled_iterator = data_scaled.as_numpy_iterator()
batch_scaled = scaled_iterator.next()

#Check to see if max is 1 and min is 0: Images between 0 and 1
'print(batch[0].min())'

#Split data:
train_size = int(len(data_scaled)*.7)
val_size = int(len(data_scaled)*.2)
test_size = int(len(data_scaled)*.1)

train = data_scaled.take(train_size)
#Skip batches we have already allocated (train_size):
val = data_scaled.skip(train_size).take(val_size)
test = data_scaled.skip(train_size+val_size).take(test_size)

#Train NN
model = Sequential([
    
    Conv2D(16, (3,3), 1, activation='relu', input_shape = (256, 256, 3)),
    MaxPooling2D(),
    Conv2D(32, (3,3), 1, activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3,3), 1, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')

])

model.compile('adam',
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])

hist = model.fit(train, epochs=5, validation_data=val)