import os
from PIL import Image
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model

import cv2

IMG_SIZE = 24

def collect():
	train_datagen = ImageDataGenerator(
			rescale=1./255,
			shear_range=0.2,
			horizontal_flip=True, 
		)

	val_datagen = ImageDataGenerator(
			rescale=1./255,
			shear_range=0.2,
			horizontal_flip=True,		)

	train_generator = train_datagen.flow_from_directory(
	    directory="dataset/train",
	    target_size=(IMG_SIZE, IMG_SIZE),
	    color_mode="grayscale",
	    batch_size=32,
	    class_mode="binary",
	    shuffle=True,
	    seed=42
	)

	val_generator = val_datagen.flow_from_directory(
	    directory="dataset/val",
	    target_size=(IMG_SIZE, IMG_SIZE),
	    color_mode="grayscale",
	    batch_size=32,
	    class_mode="binary",
	    shuffle=True,
	    seed=42
	)
	return train_generator, val_generator


def save_model(model):
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")

def load_model():
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return loaded_model

def train(train_generator, val_generator, epoch_max):
	STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
	STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

	print('[LOG] Initializing Neural Network...')
	
	model = Sequential()

	model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)))
	model.add(AveragePooling2D())

	model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
	model.add(AveragePooling2D())

	model.add(Flatten())

	model.add(Dense(units=120, activation='relu'))

	model.add(Dense(units=84, activation='relu'))

	model.add(Dense(units=1, activation = 'sigmoid'))

	print('Plotting the model...') 
	plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print('Training neural network model for %d epochs...' % (epoch_max)) 
	model.fit_generator(generator=train_generator,
	                    steps_per_epoch=STEP_SIZE_TRAIN,
	                    validation_data=val_generator,
	                    validation_steps=STEP_SIZE_VALID,
	                    epochs=epoch_max
	)
	print('Saving model to current directory...') 
	save_model(model)
	print('Done.')     


def predict(img, model):
    img = Image.fromarray(img, 'RGB').convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_arr = np.asarray(img)
    img_arr = img_arr.reshape(1,IMG_SIZE,IMG_SIZE,1)
    prediction = model.predict(img_arr)
    if prediction < 0.1:
        prediction = 'closed'
    elif prediction > 0.9:
        prediction = 'open'
    else:
        prediction = 'idk'
#     print('DEBUG - eyes state', prediction)
    return prediction


def evaluate(X_test, y_test):
	model = load_model()
	print('Evaluate model')
	loss, acc = model.evaluate(X_test, y_test, verbose = 0)
	print(acc * 100)

if __name__ == '__main__':	
	train_generator , val_generator = collect()
	train(train_generator,val_generator)
