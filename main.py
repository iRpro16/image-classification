import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os
import cv2
import imghdr

#Create dataset with images:
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