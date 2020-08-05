from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
import keras
from tqdm import tqdm

from skimage.transform import resize

from keras.layers import Conv2D, Dense, Reshape, Flatten, ReLU, Input, Dropout, BatchNormalization, MaxPooling2D
from keras.optimizers import Adam, SGD

import cv2


def load_svhn_data(dataset_path):
	SHVN_images = loadmat(dataset_path)

	images = SHVN_images['X']
	images = np.moveaxis(images, -1, 0)

	truths = SHVN_images['y']
	truths = np.where(truths==10, 0, truths)
	truths = keras.utils.to_categorical(truths, 10)

	print('loaded images and truth values.\n\nConverting to grayscale:')
	gray_images = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in tqdm(images)])
	
	print('Converted to grayscale.\n\nResizing and normalizing images:')
	norm_images = np.array([resize(im, (32, 32), mode='reflect') for im in tqdm(gray_images)])

	norm_images = np.expand_dims(norm_images, axis=3)

	return images, truths


def get_svhn_model():
	model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.3))

	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.3))

	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.3))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(10,  activation='softmax'))

	input = Input(shape=train_images[0].shape)
	output = model(input)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

	return model

def train_svhn(model, train_images, train_truths, test_images, test_truths, batch_size, num_epochs):
	history = model.fit(train_images, train_truths, batch_size=batch_size, verbose=1,
	                    validation_data=(test_images, test_truths), epochs=num_epochs)
	return history, model




train_path = "./datasets/SHVN/train_32x32.mat"
test_path = "./datasets/SHVN/test_32x32.mat"
train_images, train_truths = load_svhn_data(train_path)
test_images, test_truths = load_svhn_data(test_path)
model = get_svhn_model()

batch_size = 128
num_epochs = 20
history, model = train_svhn(model, train_images, train_truths, test_images, test_truths, batch_size, num_epochs)

# save svhn encoder:
svhn_encoder= keras.Model(inputs=model.inputs, outputs=model.layers[-3].output)
svhn_encoder.save('./svhn_encoder.h5')

# plot history:
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