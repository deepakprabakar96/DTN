from keras.layers import Input
from keras.models import load_model
from keras.models import Model
import keras

import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from scipy.io import loadmat
import cv2


def load_svhn_data(dataset_path):
	SHVN_images = loadmat(dataset_path)

	images = SHVN_images['X']
	images = np.moveaxis(images, -1, 0)

	truths = SHVN_images['y']
	truths = np.where(truths==10, 0, truths)
	truths = keras.utils.to_categorical(truths, 10)

	print('loaded images and truth values.')
	gray_images = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in images])
	
	print('Converted to grayscale.\nResizing and normalizing images:')
	norm_images = np.array([resize(im, (32, 32), mode='reflect') for im in tqdm(gray_images)])

	norm_images = np.expand_dims(norm_images, axis=3)

	return norm_images, truths


def get_dtn(encoder_model_path, decoder_g_path):
	encoder_f = load_model(encoder_model_path)
	decoder_g = load_model(decoder_g_path)
	inp = Input(shape=(32,32,1))
	encoded_op = encoder_f(inp)
	dtn_op = decoder_g(encoded_op)
	dtn = Model(inputs=inp, outputs=dtn_op)
	return dtn



test_path = "./test_32x32.mat"
test_images, test_truths = load_svhn_data(test_path)

encoder_model_path = './svhn_encoder.h5'
decoder_g_path = './model/generator_2800.h5'
dtn = get_dtn(encoder_model_path, decoder_g_path)

mnist_model = load_model('./mnist_clf.h5')

test_preds = []
for i in tqdm(range(0,len(test_images),100)):
	dtn_op = dtn.predict(test_images[i:i+100])
	test_preds.extend(mnist_model.predict(dtn_op))

preds = np.array([np.argmax(i) for i in test_preds])
truths = np.array([np.argmax(i) for i in test_truths])

print('accuracy: {}'.format(np.where(preds==truths)[0].shape[0]/preds.shape[0]))