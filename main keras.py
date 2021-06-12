#import various packages
import os
import numpy as np
import imageio

import keras
from keras.models import Sequential
import cv2
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#Defining the File Path
datasetFilepath ="./dataset"
dataset=os.listdir(datasetFilepath)
SIZE=128


#Loading the Images
images = []
label = []
labelNames = []
for i,f in enumerate(dataset):
    print(i,f)
    labelNames.append(f)
    folderPath=datasetFilepath+"/"+f
    folder = os.listdir(folderPath)
    for j in folder:
        image = imageio.imread(folderPath+'/'+j)
        if(len(image.shape)==3):
            images.append(image/255)
            label.append(i)


#resizing all the images
for i in range(len(images)):
    images[i]=cv2.resize(images[i],(SIZE,SIZE))



#converting images to arrays
images=np.array(images)
label=np.array(label)
print(images.shape)

X_train, X_test, y_train, y_test = train_test_split(images, label, test_size = 0.15, random_state = 0)
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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
history=model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize,validation_split=0.15)

# model accuracy
_, acc = model.evaluate(X_test, y_test)
print("Accuracy is = ", (acc * 100.0), "%") 

model.summary()


# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# save model
model.save('model_1.h5')

