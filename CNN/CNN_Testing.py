import numpy as np
import pickle
import cv2
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from os import listdir
import keras
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

width = 256
height = 256
depth = 3

inputShape = (height, width, depth)

if os.path.exists('cnn_model.pkl'):
    with open("cnn_model.pkl", "rb") as fo:
        model = pickle.load(fo)

if os.path.exists('label_transform.pkl'):
    with open("label_transform.pkl", "rb") as fo:
        label_binarizer = pickle.load(fo)

all_labels = label_binarizer.classes_
print(all_labels)

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
im = keras.preprocessing.image.load_img(file_path, target_size=inputShape)


doc = keras.preprocessing.image.img_to_array(im) # -> numpy array
print(type(doc), doc.shape)
# we can use a numpy routine to create an axis in the first position
doc = np.expand_dims(doc, axis=0)
print(type(doc), doc.shape)

prediction = model.predict_classes(doc)

print(prediction)

prediction_label = all_labels[prediction]
print(prediction_label[0])
