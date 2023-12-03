import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
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


