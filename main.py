import copy
import os
import pickle
import sys
from itertools import chain
from os import listdir
from subprocess import call

# Importing Keras libraries and packages
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K #Tensorflow Backend
from keras.applications.vgg16 import VGG16
from keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import array_to_img, img_to_array
from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication, QTimer, pyqtSlot
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

# List of 38 classes of healthy and diseased plants
CLASSES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
      'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
      'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 
      'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 
      'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


class PlantDiseaseClassification(QMainWindow):
    
    def __init__(self):
        super(PlantDiseaseClassification, self).__init__()
        
        loadUi("MainWindow_Gui.ui", self)
        
        self.TrainButton.clicked.connect(self.TrainButtonSlotFunction) #Signal Function
        self.AccuracyButton.clicked.connect(self.AccuracyButtonSlotFunction) #Signal Function
        self.BrowseButton.clicked.connect(self.BrowseButtonSlotFunction)
        self.PredictionButton.clicked.connect(self.PredictionButtonSlotFunction)
        self.ExitButton.clicked.connect(self.ExitButtonSlotFunction)
        
        self.algo_names = ['CNN', 'Alexnet', 'VGG16', 'Resnet50']        
        self.AlgoNameComboBox.addItems(self.algo_names)
        self.selected_algo = str(self.AlgoNameComboBox.currentText())
        print(self.selected_algo)
        self.qm = QMessageBox()
        self.image_list, self.label_list = [], []
        self.default_image_size = tuple((256, 256))
        
    @pyqtSlot() #Train Process
    def TrainButtonSlotFunction(self):
        self.selected_algo = str(self.AlgoNameComboBox.currentText())
        
        if self.selected_algo == 'CNN':            
            ret = self.qm.question(self,'', "The Training process may take 4 to 12 hours,\ndepending on your system configuration.\n\nDo you really want to continue.", self.qm.Yes | self.qm.No)
            if ret == self.qm.Yes:               
                self.CNN_Training()
            else:
                print("Training Avoided !")
                
        elif self.selected_algo == 'Alexnet':            
            ret = self.qm.question(self,'', "The Training process may take 4 to 12 hours,\ndepending on your system configuration.\n\nDo you really want to continue.", self.qm.Yes | self.qm.No)
            if ret == self.qm.Yes:               
                self.AlexNet_Training()
            else:
                print("Training Avoided !")
                
        elif self.selected_algo == 'VGG16':            
            ret = self.qm.question(self,'', "The Training process may take 4 to 12 hours,\ndepending on your system configuration.\n\nDo you really want to continue.", self.qm.Yes | self.qm.No)

            if ret == self.qm.Yes:               
                self.VGG16_Training()
            else:
                print("Training Avoided !")
                
        elif self.selected_algo == 'Resnet50':            
            ret = self.qm.question(self,'', "The Training process may take 4 to 12 hours,\ndepending on your system configuration.\n\nDo you really want to continue.", self.qm.Yes | self.qm.No)

            if ret == self.qm.Yes:               
                self.VGG16_Training()
            else:
                print("Training Avoided !")
    
    def CNN_Training(self):
        print("CNN_Training !")
        call(["python", ".\\CNN\\CNN_Training.py"])
    
    def AlexNet_Training(self):
        print("AlexNet_Training !")
        call(["python", ".\\Alexnet\\Alexnet_Training.py"])
            
    def VGG16_Training(self):
        print("VGG16_Training !")
        call(["python", ".\\VGG16\\VGG16_Training.py"])
        
    def Resnet50_Training(self):
        print("VGG16_Training !")
        call(["python", ".\\Resnet50\\RESNET50_Training.py"])
        
    
    @pyqtSlot() #Accuracy & plot/Slot function
    def AccuracyButtonSlotFunction(self):
        self.selected_algo = str(self.AlgoNameComboBox.currentText())
        print(self.selected_algo)
        
        if self.selected_algo == 'CNN':      
            self.Accuracy_label.setText('98.95')                  
            call(["python", ".\\CNN\\CNN_accuracy.py"])
            
        elif self.selected_algo == 'Alexnet':       
            self.Accuracy_label.setText('96.89')                 
            call(["python", ".\\Alexnet\\Alexnet_accuracy.py"])
            
        elif self.selected_algo == 'VGG16':      
            self.Accuracy_label.setText('98.05')                  
            call(["python", ".\\VGG16\\VGG16_accuracy.py"])
            
        elif self.selected_algo == 'Resnet50':      
            self.Accuracy_label.setText('98.24')                  
            call(["python", ".\\Resnet50\\RESNET50_accuracy.py"])
            
    
    @pyqtSlot() #Browse Image Process
    def BrowseButtonSlotFunction(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File', '.\\', "Image Files (*.*)")
        if fname:
            self.LoadImageFunction(fname)
        else:
            print("No Valid File selected.")
    
    def LoadImageFunction(self, fname):
        self.image = cv2.imread(fname)
        self.DisplayImage(self.image, 1)

    def DisplayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)

        outImg = outImg.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImg))
            self.imgLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)
    
    @pyqtSlot() #Testing Process
    def PredictionButtonSlotFunction(self):
        self.selected_algo = str(self.AlgoNameComboBox.currentText())
        print(self.selected_algo)
        
        if self.selected_algo == 'CNN':                        
            self.CNN_Prediction(self.image)                               # call(["python", ".\\CNN\\CNN_Testing.py"])     
        elif self.selected_algo == 'Alexnet':                        
            self.AlexNet_Prediction(self.image)                           # call(["python", ".\\Alexnet\\Alexnet_Testing.py"])
        elif self.selected_algo == 'VGG16':                        
            self.VGG16_Prediction(self.image)                             # call(["python", ".\\VGG16\\VGG16_Testing.py"])
        elif self.selected_algo == 'Resnet50':                        
            self.Resnet50_Prediction(self.image)                          # call(["python", ".\\Resnet50\\RESNET50_Testing.py"])
    
    def CNN_Prediction(self, img):
        width = 256
        height = 256
        depth = 3

        inputShape = (height, width, depth)

        if os.path.exists('.\\CNN\\cnn_model.pkl'):
            with open(".\\CNN\\cnn_model.pkl", "rb") as fo:
                model = pickle.load(fo)

        if os.path.exists('.\\CNN\\label_transform.pkl'):
            with open(".\\CNN\\label_transform.pkl", "rb") as fo:
                label_binarizer = pickle.load(fo)

        all_labels = label_binarizer.classes_
        # print(all_labels)

        # pre-process the image for classification
        image = cv2.resize(img, (height, width))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)             

        prediction = model.predict_classes(image)

        print(prediction)

        prediction_label = all_labels[prediction]
        print(prediction_label[0])
        
        self.qm.information(self,'Prediction Result', str(prediction_label[0]), self.qm.Close)
    
    def AlexNet_Prediction(self, img):
        model = self.AlexNet_Load_Model()
        
        # pre-process the image for classification
        image = cv2.resize(img, (224, 224))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        print(predictions)
        
        d = predictions.flatten()
        j = d.max()
        for index,item in enumerate(d):
            if item == j:
                class_name = CLASSES[index]

        print('\n\n')
        print(class_name)
        
        self.qm.information(self,'Prediction Result', str(class_name), self.qm.Close)        
    
    def AlexNet_Load_Model(self):
        classifier = Sequential()
        
        classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
        classifier.add(BatchNormalization())
        classifier.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
        classifier.add(BatchNormalization())
        classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
        classifier.add(BatchNormalization())
        classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
        classifier.add(BatchNormalization())
        classifier.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
        classifier.add(BatchNormalization())
        classifier.add(Flatten())
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
        # classifier.summary()
        
        for i, layer in enumerate(classifier.layers[:20]):
            print(i, layer.name)
            layer.trainable = False
        
        classifier.load_weights('.\\Alexnet\\AlexNetModel.hdf5')
        
        return classifier
    
    def VGG16_Prediction(self, img):
        
        model = self.VGG16_Load_Model()
        
        # pre-process the image for classification
        image = cv2.resize(img, (224, 224))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        print(predictions)
        
        predicted_class_name = CLASSES[np.argmax(predictions)]
        print("Detected the leaf as ", predicted_class_name) 
        
        self.qm.information(self,'Prediction Result', str(predicted_class_name), self.qm.Close)        
    
    def VGG16_Load_Model(self):         
        
        in_shape=(224, 224, 3)
        out_shape=38  
        
        model = VGG16(include_top=False, input_shape=in_shape, weights=".\\VGG16\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
        
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
    
        model.load_weights('.\\VGG16\\VGG16_bestweights.hdf')
    
        return model    
    
    def Resnet50_Prediction(self, img):
        
        model = self.Resnet50_Load_Model()
        
        # pre-process the image for classification
        image = cv2.resize(img, (256, 256))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        print(predictions)
        
        predicted_class_name = CLASSES[np.argmax(predictions)]
        print("Detected the leaf as ", predicted_class_name) 
        
        self.qm.information(self,'Prediction Result', str(predicted_class_name), self.qm.Close)                
    
    def Resnet50_Load_Model(self):
        
        img_width, img_height = 256, 256

        resnet50 = ResNet50(include_top=False, weights="imagenet", input_shape=(img_width, img_height, 3))
        resnet50.trainable = False

        x = resnet50.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(38, activation="softmax")(x)

        model = Model(resnet50.input, x)
        model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        model.load_weights('.\\ResNet50\\ResNet50_model.h5')
        
        return model
    
    @pyqtSlot()
    def ExitButtonSlotFunction(self):
        QApplication.instance().quit()
                
    ''' ------------------------ MAIN Function ------------------------- '''
       
    
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    window = PlantDiseaseClassification()
    window.setWindowTitle('Plant Disease Classification using Pretrained Deep Learning Algorithm')
    window.show()
    sys.exit(app.exec_())
