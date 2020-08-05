'''
Training mnist on normalized 32x32 images
'''

from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
import keras
from tqdm import tqdm

from skimage.transform import resize

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Reshape, Flatten, ReLU, Input, Dropout, BatchNormalization, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.datasets import mnist

import cv2


def load_mnist_data():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	y_train = np.where(y_train==10, 0, y_train)
	y_train = keras.utils.to_categorical(y_train, 10)
	y_test = np.where(y_test==10, 0, y_test)
	y_test = keras.utils.to_categorical(y_test, 10)

	print('Resizing images:')
	X_train = np.array([resize(im, (32, 32), mode='reflect') for im in tqdm(X_train)])
	X_test = np.array([resize(im, (32, 32), mode='reflect') for im in tqdm(X_test)])

	X_train = np.expand_dims(X_train, axis=3)
	X_test = np.expand_dims(X_test, axis=3)

	return X_train, y_train, X_test, y_test


def get_mnist_model():
	model=Sequential()

	model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(32,32,1)))
	model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))

	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())
	model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
	model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))

	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())    
	model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
	    
	model.add(MaxPooling2D(pool_size=(2,2)))
	    
	model.add(Flatten())
	model.add(BatchNormalization())
	model.add(Dense(512,activation="relu"))
	    
	model.add(Dense(10,activation="softmax"))
	    
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

	return model

def train_mnist(model, train_images, train_truths, test_images, test_truths, batch_size, num_epochs):
	history = model.fit(train_images, train_truths, batch_size=batch_size, verbose=1,
	                    validation_data=(test_images, test_truths), epochs=num_epochs)
	return history, model


train_images, train_truths, test_images, test_truths = load_mnist_data()
model = get_mnist_model()
batch_size = 128
num_epochs = 20
history, model = train_mnist(model, train_images, train_truths, test_images, test_truths, batch_size, num_epochs)

model.save('./mnist_clf.h5')

fig, ax = plt.subplots(1,2,True,figsize=(16,8))
ax[0].plot(history.history['accuracy'], label='Train Acc')
ax[0].plot(history.history['val_accuracy'], label='Validation Acc')
ax[0].set_title("Accuracies")
ax[0].legend()
ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Validation Loss')
ax[1].set_title("Losses")
ax[1].legend()
plt.show()