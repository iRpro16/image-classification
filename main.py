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

#Create data directory with images:
data_dir = 'PetImages/Train'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

#Remove any unecessary extensions
for image_class in os.listdir(data_dir):
    #Loop through every single image in sub-directories
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list ()'.format(image_path))
                #Deletes a file
                os.remove(image_path)
        except Exception as e:
            print('Issue with image ()'.format(image_path))

#Load Data
tf.data.Dataset

#Builds data pipeline
data = tf.keras.utils.image_dataset_from_directory('PetImages/Train')

#Allows us to access data pipeline
data_iterator = data.as_numpy_iterator()

#Accessing data pipeline itself
batch = data_iterator.next()

#Class 1 = Dog
#Class 0 = Cat
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

#2.Preprocess data

#2.1 Scale Data
data = data.map(lambda x,y: (x/255, y))


