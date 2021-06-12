#import various packages
import os
import numpy as np
import imageio

import keras
from keras.models import Sequential
import cv2


#Defining the File Path
datasetFilepath ="./dataset"
dataset=os.listdir(datasetFilepath)
SIZE=128


#Loading the Images
label = []
labelNames = []
for i,f in enumerate(dataset):
    print(i,f)
    labelNames.append(f)
    label.append(i)


# Defining the hyperparameters
filters = 32
filtersize = (5, 5)
epochs = 100
batchsize = 128
input_shape = (SIZE, SIZE, 3)
strides=(1,1)
padding='same'
activation='tanh'
pool_size=(5, 5)
rate=0.2

#Defining the model
model = Sequential()
model.add(keras.layers.InputLayer(input_shape=input_shape))

model.add(keras.layers.convolutional.Conv2D(filters, filtersize, strides, padding, data_format="channels_last", activation=activation))
model.add(keras.layers.MaxPooling2D(pool_size))
model.add(keras.layers.BatchNormalization(axis=-1))
model.add(keras.layers.Dropout(rate))

model.add(keras.layers.convolutional.Conv2D(filters, filtersize, strides, padding, data_format="channels_last", activation=activation))
model.add(keras.layers.MaxPooling2D(pool_size))
model.add(keras.layers.BatchNormalization(axis=-1))
model.add(keras.layers.Dropout(rate))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(512,activation=activation))
model.add(keras.layers.BatchNormalization(axis=-1))
model.add(keras.layers.Dropout(rate))

model.add(keras.layers.Dense(256,activation=activation))
model.add(keras.layers.BatchNormalization(axis=-1))
model.add(keras.layers.Dropout(rate))

model.add(keras.layers.Dense(len(labelNames),activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# load model
model.load_weights('model.h5')



# choose image name and see prediction label name
test_img=np.array(cv2.resize(imageio.imread('./kangaro.jpg')/255,(SIZE,SIZE)))
test_img_input=np.expand_dims(test_img, 0)
prediction = model.predict(test_img_input)[0]
prediction= np.argmax(prediction)

print('Image category', "kangaroo")
print('prediction label',prediction)
print('label Name', labelNames[prediction])
