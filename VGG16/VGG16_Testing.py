import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sys
import os
import pickle
from numpy import load
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras import backend
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization

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


def Load_Training_Model(in_shape=(224, 224, 3), out_shape=38):
    # load model
	# model = VGG16(include_top=False, input_shape=in_shape, weights=".\\saved_models\\VGG16_bestweights.hdf")
 
	model = VGG16(include_top=False, input_shape=in_shape, weights=".\\pretrained_models\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
	#model.load_weights('../input/VGG-16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    # mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	
	
	# allow last vgg block to be trainable
	model.get_layer('block5_conv1').trainable = True
	model.get_layer('block5_conv2').trainable = True
	model.get_layer('block5_conv3').trainable = True
	model.get_layer('block5_pool').trainable = True
	
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	fcon1 = Dense(4096, activation='relu', kernel_initializer='he_uniform')(flat1)
	fdrop1 = Dropout(0.25)(fcon1)
	fbn1 = BatchNormalization()(fdrop1)
	fcon2 = Dense(4096, activation='relu', kernel_initializer='he_uniform')(fbn1)
	fdrop2 = Dropout(0.25)(fcon2)
	fbn2 = BatchNormalization()(fdrop2)
	output = Dense(out_shape, activation='softmax')(fbn2)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.01, momentum=0.9,decay=0.005)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
 
	model.load_weights('.\\saved_models\\VGG16_bestweights.hdf')
 
	return model

def Predict_Test_Image_File(model):
    root = tk.Tk()
    root.withdraw()
    imageFileName = filedialog.askopenfilename()

    image = cv2.imread(imageFileName)
        
    # pre-process the image for classification
    image = cv2.resize(image, (224, 224))
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
        

