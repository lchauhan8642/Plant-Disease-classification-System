#Code Reference: https://www.kaggle.com/srikanthreddyt/plant-disease-classification-with-pretrained-vgg16

# Importing Keras libraries and packages
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

# -------------------------------------------------------------------------------------------

traindir = "./dataset/train"
validdir = "./dataset/valid"
testdir = "./test/test"

# -------------------------------------------------------------------------------------------

# define cnn model
def define_model(in_shape=(224, 224, 3), out_shape=38):
	# load model
	model = VGG16(include_top=False, input_shape=in_shape, weights="./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")  #https://www.kaggle.com/yuhh1989/vgg16-weights
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
	return model



# -------------------------------------------------------------------------------------------



batch_size = 128

train_datagen = ImageDataGenerator(rescale=1./255)
                                   
valid_datagen = ImageDataGenerator(rescale=1./255)

training_iterator = train_datagen.flow_from_directory(traindir,
                                                 target_size=(224, 224),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')

test_iterator = valid_datagen.flow_from_directory(validdir,
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            class_mode='categorical')

class_dict = training_iterator.class_indices
print(class_dict)

class_labels = list(class_dict.keys())
print(class_labels)

train_num_samples = training_iterator.samples
valid_num_samples = test_iterator.samples
# define model
model = define_model()
model.summary()

weightsfilepath = ".\\pretrained_models\\bestweights_vgg16.hdf5"
checkpoint = ModelCheckpoint(weightsfilepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [checkpoint]

# fit model
history = model.fit_generator(training_iterator, 
                              steps_per_epoch=train_num_samples//batch_size,
                              validation_data=test_iterator,
                              validation_steps=valid_num_samples//batch_size, 
                              epochs=15, 
                              callbacks=callbacks_list, 
                              verbose=2)

# save the train history
with open("vgg16_train_history.pkl", 'wb') as f:
    pickle.dump(history, f)


with open("VGG16_train_history.pkl", 'wb') as f:
    pickle.dump(history, f)


model.save('.\\saved_models\\VGG16_bestweights.hdf5')