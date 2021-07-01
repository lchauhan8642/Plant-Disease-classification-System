#Code Reference: https://github.com/imskr/Plant_Disease_Detection/blob/master/notebook/Plant_Disease_RESNET50.ipynb

# Importing Keras libraries and packages
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
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


train_data_dir = ".\\dataset\\train"
validation_data_dir = ".\\dataset\\valid"

img_width, img_height = 256, 256
batch = 20
epochs = 2



class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)



# -------------------------------------------------------------------------------------------


# ImageDataGenerator

train_datagen = ImageDataGenerator(
                                        horizontal_flip=True,
                                        zoom_range=0.1,
                                        rotation_range=10,
                                        preprocessing_function=preprocess_input
                                    )

val_datagen = ImageDataGenerator(
                                    horizontal_flip=True,
                                    zoom_range=0.1,
                                    rotation_range=10, 
                                    preprocessing_function=preprocess_input
                                )

# train, validation datagen

train_generator = train_datagen.flow_from_directory(
                                                        directory=train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch,
                                                    )

val_generator = val_datagen.flow_from_directory(
                                                    directory=validation_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch,
                                                )


# -------------------------------------------------------------------------------------------

# Model definition

resnet50 = ResNet50(include_top=False, weights="imagenet", input_shape=(img_width, img_height, 3))
resnet50.trainable = False

x = resnet50.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(38, activation="softmax")(x)


model = Model(resnet50.input, x)

# model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy', f1, precision, recall])
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("res_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


time_callback = TimeHistory()

train_num = train_generator.samples
valid_num = val_generator.samples

print(train_num)
print(valid_num)

# -------------------------------------------------------------------------------------------

# Model Training


history = model.fit_generator(
                                train_generator,
                                epochs=5,
                                steps_per_epoch=1000,
                                validation_data=val_generator,
                                validation_steps=500,
                                callbacks = [checkpoint, early, time_callback]
                            )


print(time_callback.times)

# -------------------------------------------------------------------------------------------


with open('ResNet50_model_history.pkl','wb') as f:
    pickle.dump(history, f)

with open('ResNet50_model_epoch_times.pkl','wb') as fa:
    pickle.dump(time_callback.times, fa)


model.save('ResNet50_model.h5')

# -------------------------------------------------------------------------------------------

