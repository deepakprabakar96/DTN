'''
Implementation without L_const, L_tid and L_tv:
Need to update function load_bitmojis()
Need to update function load_faces()
Need to add code for normalizing face and bitmoji images in the function train()
Need to update code to include L_const, L_tid
'''

from facenet.preprocessing import align_images

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers import Conv2D
from keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm


def load_bitmojis():
	# update later

def load_faces():
	# update later


class DTN:
	def __init__(self, facedet_cascade_path, facenet_model_path):
		self.img_rows = 160
		self.img_cols = 160
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)

		self.optimizer = Adam(0.0002, 0.5)

		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

		self.cascade_facedet = cv2.CascadeClassifier(facedet_cascade_path)
		self.encoder_f = load_model(facenet_model_path)
		self.encoder_f.trainable = False

		self.decoder_g = self.build_decoder_g()
		self.dtn = Model()
		self.build_dtn()

	def build_discriminator(self):

		init = RandomNormal(stddev=0.02)
		model = Sequential()

		model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=self.img_shape))
		model.add(LeakyReLU(alpha=0.2))

		model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
		model.add(LeakyReLU(alpha=0.2))

		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))

		opt = Adam(lr=0.0002, beta_1=0.5)
		model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

		return model

	def encoder_preprocess(self, image):
		aligned_image = align_images(self.cascade_facedet, image)
		return aligned_image

	def build_decoder_g(self):
		init = RandomNormal(stddev=0.02)
		model = Sequential()

		n_nodes = 128 * 56 * 56
		model.add(Dense(n_nodes, kernel_initializer=init, input_dim=28*28*8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Reshape((56, 56, 128)))
		# 128x128:
		model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
		model.add(LeakyReLU(alpha=0.2))
		# 224x224:
		model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
		model.add(LeakyReLU(alpha=0.2))
		# 224x224x3:
		model.add(Conv2D(3, (7,7), activation='tanh', padding='same', kernel_initializer=init))
		return model

	def build_dtn(self):
		inp = Input(shape=self.img_shape)
		encoded_op = self.encoder_f(inp)
		generator_op = self.decoder_g(encoded_op)

		self.discriminator.trainable = False

		discriminator_op = self.discriminator(generator_op)

		self.dtn = Model(inp, discriminator_op)
		self.dtn.compile(loss='binary_crossentropy', optimizer=self.optimizer)

	def train(self, epochs, batch_size):

		bitmoji_imgs = load_bitmojis()
		faces = load_faces()

		# Important!! :
		# Normalize bitmoji_imgs: update later
		# Normalize faces: update later

		y_1 = np.zeros((batch_size,3))
		y_1[:,2] = np.ones(batch_size)
		y_2 = np.zeros((batch_size,3))
		y_2[:,1] = np.ones(batch_size)
		y_3 = np.zeros((batch_size,3))
		y_3[:,0] = np.ones(batch_size)

		y_dtn = np.concatenate((y_3,y_3))

		for epoch in range(epochs):
			for batch in tqdm(range(int(faces.shape[0]/batch_size))):
				idx = np.random.randint(0, bitmoji_imgs.shape[0], batch_size)
				org_imgs = bitmoji_imgs[idx]

				idx = np.random.randint(0, faces.shape[0], batch_size)
				x_S = faces[idx]

				f_x_S = self.encoder_f.predict(x_S)
				f_x_T = self.encoder_f.predict(org_imgs)

				g_x_S = self.decoder_g.predict(f_x_S)
				g_x_T = self.decoder_g.predict(f_x_T)

				self.discriminator.trainable = True
				L_D1, acc_D1 = self.discriminator.train_on_batch(g_x_S, y_1)
				L_D2, acc_D2 = self.discriminator.train_on_batch(g_x_T, y_2)
				L_D3, acc_D3 = self.discriminator.train_on_batch(org_imgs, y_3)

				L_D = L_D1 + L_D2 + L_D3
				acc_D = (acc_D1 + acc_D2 + acc_D3)/3

				x_dtn = np.concatenate((g_x_S,g_x_T))

				self.discriminator.trainable = False
				L_GANG = self.dtn.train_on_batch(x_dtn, y_dtn)

				# L_const = MSE ( )?
				# L_TID = MSE ( )?

				print("epoch: "+ str(epoch)+ ", batch_count: "+str(batch) + ", L_D: "+ str(L_D)+ ", L_GANG: "+ str(L_GANG)+ ", accuracy:"+ str(acc_D))


dtn = DTN()
dtn.train(epochs=100, batch_size=64)