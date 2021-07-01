import h5py
import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import img_to_array, array_to_img, load_img

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2

# List of 38 classes of healthy and diseased plants
li = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
      'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
      'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 
      'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 
      'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

def Load_Training_Model():
    # Initializing the CNN
    classifier = Sequential()

    # Convolution Step 1
    classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))

    # Max Pooling Step 1
    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
    classifier.add(BatchNormalization())

    # Convolution Step 2
    classifier.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))

    # Max Pooling Step 2
    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
    classifier.add(BatchNormalization())

    # Convolution Step 3
    classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
    classifier.add(BatchNormalization())

    # Convolution Step 4
    classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
    classifier.add(BatchNormalization())

    # Convolution Step 5
    classifier.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))

    # Max Pooling Step 3
    classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
    classifier.add(BatchNormalization())

    # Flattening Step
    classifier.add(Flatten())

    # Full Connection Step
    classifier.add(Dense(units = 4096, activation = 'relu'))
    classifier.add(Dropout(0.4))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units = 4096, activation = 'relu'))
    classifier.add(Dropout(0.4))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units = 1000, activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units = 38, activation = 'softmax'))
    classifier.summary()
    
    for i, layer in enumerate(classifier.layers[:20]):
        print(i, layer.name)
        layer.trainable = False
    
    classifier.load_weights('.\\saved_models\\AlexNetModel.hdf5')
    
    return classifier
    
        
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
    
    d = predictions.flatten()
    j = d.max()
    for index,item in enumerate(d):
        if item == j:
            class_name = li[index]

    print('\n\n')
    print(class_name)
    
    tk.messagebox.showinfo('Test Image Prediction',class_name)  
    
    # final_prediction = predictions.argmax(axis=1)[0]

    # print(final_prediction)
    
    
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
        

