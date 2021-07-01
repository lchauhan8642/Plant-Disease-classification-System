import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.optimizers import Adam
from keras import backend as K
import time
import pickle
from keras.models import load_model

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2

# List of 38 classes of healthy and diseased plants
class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 
                'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 
                'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_m = true_positives / (possible_positives + K.epsilon())
    return recall_m

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_m = true_positives / (predicted_positives + K.epsilon())
    return precision_m

def f1(y_true, y_pred):
    precision_m = precision(y_true, y_pred)
    recall_m = recall(y_true, y_pred)
    return 2*((precision_m*recall_m)/(precision_m+recall_m+K.epsilon()))


# -------------------------------------------------------------------------------------------

def Load_Training_Model():

    img_width, img_height = 256, 256
    batch = 20
    epochs = 2

    resnet50 = ResNet50(include_top=False, weights="imagenet", input_shape=(img_width, img_height, 3))
    resnet50.trainable = False

    x = resnet50.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(38, activation="softmax")(x)

    model = Model(resnet50.input, x)
    # model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy', f1, precision, recall])
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights('ResNet50_model.h5')
    # model.summary()
    
    return model

def Predict_Test_Image_File(model):
    root = tk.Tk()
    root.withdraw()
    imageFileName = filedialog.askopenfilename()

    image = cv2.imread(imageFileName)
        
    # pre-process the image for classification
    image = cv2.resize(image, (256, 256))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    print(predictions)
    
    predicted_class_name = class_labels[np.argmax(predictions)]
    print("Detected the leaf as ", predicted_class_name) 
    
    tk.messagebox.showinfo('Test Image Prediction',predicted_class_name)  
    

# ----------------------------------------- MAIN FUNCTION ------------------------------------------------------

if __name__ == "__main__":
        
    model_definition = Load_Training_Model()
    
    # print(model_definition.history)
    
    root= tk.Tk() # create window
    root.withdraw()

    MsgBox = tk.messagebox.askquestion ('Tensorflow Predictions','Do you want to test Images for Predictions')
    
    while MsgBox == 'yes':
        MsgBox = tk.messagebox.askquestion ('Test Image','Do you want to test new Image')
        if MsgBox == 'yes':
            Predict_Test_Image_File(model_definition)            
        else:
            tk.messagebox.showinfo('EXIT', "Exiting the Application")
            break
        

