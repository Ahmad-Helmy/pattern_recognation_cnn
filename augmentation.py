# example of brighting image augmentation
from operator import le
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
import os
import imageio
import cv2
from PIL import Image




def augmented_brightness(image_array,path):
	# expand dimension to one sample
	samples = expand_dims(image_array, 0)
	# create image data augmentation generator
	datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
	# prepare iterator
	it = datagen.flow(samples, batch_size=1)
	# generate samples and plot
	for i in range(9):
		# generate batch of images
		batch = it.next()
		# convert to unsigned integers for viewing
		image = batch[0].astype('uint8')
		im = Image.fromarray(image)
		im.save(path+"brightness%d.jpg"%(i+1))


def augmented_zoom(image_array,path):
	samples = expand_dims(image_array, 0)
	# create image data augmentation generator
	datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
	# prepare iterator
	it = datagen.flow(samples, batch_size=1)
	# generate samples and plot
	for i in range(9):
		# generate batch of images
		batch = it.next()
		# convert to unsigned integers for viewing
		image = batch[0].astype('uint8')
		im = Image.fromarray(image)
		im.save(path+"zoom%d.jpg"%(i+1))

def augmented_rotate(image_array,path):
	# rotate img 90 deg
	im=cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
	cv2.imwrite(path+"rotate%d.jpg"%(90), im)
	# rotate img 180 deg
	im=cv2.rotate(image_array, cv2.ROTATE_180)
	cv2.imwrite(path+"rotate%d.jpg"%(180), im)
	# rotate img 270 deg
	im=cv2.rotate(image_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
	cv2.imwrite(path+"rotate%d.jpg"%(270), im)

def augmented_translation(image_array,path):
	
	samples = expand_dims(image_array, 0)
	# create image data augmentation generator
	datagen = ImageDataGenerator(height_shift_range=0.35,width_shift_range=[-200,200], fill_mode='constant')
	# prepare iterator
	it = datagen.flow(samples, batch_size=1)
	# generate samples and plot
	for i in range(9):
		# generate batch of images
		batch = it.next()
		# convert to unsigned integers for viewing
		image = batch[0].astype('uint8')
		im = Image.fromarray(image)
		im.save(path+"translate%d.jpg"%(i+1))



#Defining the File Path
datasetFilepath ="./dataset"
dataset=os.listdir(datasetFilepath)

#Loading the Images
for i,f in enumerate(dataset):
	folderPath=datasetFilepath+"/"+f
	folder = os.listdir(folderPath)
	for j in folder:
		image = imageio.imread(folderPath+'/'+j)
		imgPath=folderPath+'/'+j.split('.')[0]
		if(len(image.shape)==3):
			augmented_brightness(image,imgPath)
			augmented_zoom(image,imgPath)
			augmented_rotate(image,imgPath)


