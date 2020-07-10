'''
Need to update function load_bitmojis()
Need to update function load_faces()
Need to add code for normalizing face and bitmoji images in the function train()
Change encoded_op_shape
Change facedet_cascade_path, facenet_model_path
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


def load_bitmojis(batch_size):
	# generate bitmoji images for 1 batch
	# reshape to (160,160,3)
	# update later
	return np.array([np.zeros((160,160,3))]*batch_size)	# temporary

def load_faces(batch_size):
	# generate face images for 1 batch
	# reshape to (160,160,3)
	# update later
	return np.array([np.zeros((160,160,3))]*batch_size)	# temporary


class DTN:
	def __init__(self, facedet_cascade_path, facenet_model_path):
		self.img_rows = 160
		self.img_cols = 160
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)

		self.optimizer = Adam(0.0002, 0.5)

		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

		self.cascade_facedet = cv2.CascadeClassifier(facedet_cascade_path)
		self.encoder_f = load_model(facenet_model_path)
		self.encoder_f.trainable = False

		self.decoder_g = self.build_decoder_g()

		self.discriminator.trainable = False
		self.build_dtn()

	def build_discriminator(self):

		init = RandomNormal(stddev=0.02)
		model = Sequential()

		model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=self.img_shape))
		model.add(LeakyReLU(alpha=0.2))

		model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
		model.add(LeakyReLU(alpha=0.2))

		model.add(Flatten())
		model.add(Dense(3, activation='softmax'))

		return model

	def encoder_preprocess(self, image):
		aligned_image = align_images(self.cascade_facedet, image)
		return aligned_image

	def build_decoder_g(self):
		init = RandomNormal(stddev=0.02)
		model = Sequential()

		n_nodes = 128 * 40 * 40
		encoded_op_shape = (0, 0, 0)
		model.add(Dense(n_nodes, kernel_initializer=init, input_dim=encoded_op_shape))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Reshape((40, 40, 128)))
		# 80x80x128:
		model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
		model.add(LeakyReLU(alpha=0.2))
		# 160x160x128:
		model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
		model.add(LeakyReLU(alpha=0.2))
		# 160x160x3:
		model.add(Conv2D(3, (7,7), activation='tanh', padding='same', kernel_initializer=init))
		return model

	def L_const(y_true, y_pred):
		'''
		keras loss function expects only y_true and y_pred as params.
		While training on a batch, we pass (source, y_true) for the param y_true 
		so that we can optimize weights only when x ∈ s.
		source=1 if x ∈ s, source=0 if x ∈ t
		'''
		source, y_t = y_true
		if source == 1:
			return (y_t - y_pred)**2
		else:
			return 0		# if y_true = y_pred, loss is 0 and weights won't get updated

	def L_tid(y_true, y_pred):
		'''
		keras loss function expects only y_true and y_pred as params.
		While training on a batch, we pass (source, y_true) for the param y_true 
		so that we can optimize weights only when x ∈ t.
		source=1 if x ∈ t, source=0 if x ∈ s
		'''
		source, y_t = y_true
		if source == 1:
			return (y_t - y_pred)**2
		else:
			return 0		# if y_true = y_pred, loss is 0 and weights won't get updated

	def build_dtn(self):
		alpha = 100
		beta = 1

		inp = Input(shape=self.img_shape)
		encoded_op = self.encoder_f(inp)
		generator_op = self.decoder_g(encoded_op)

		discriminator_op = self.discriminator(generator_op)

		encoded_op2 = self.encoder_f(generator_op)

		self.dtn = Model(inputs=inp, outputs=[discriminator_op, encoded_op2, generator_op])

		losses = {'L_GANG':'categorical_crossentropy','L_const':L_const,'L_tid':L_tid}
		lossWeights = {'L_GANG':1,'L_const':alpha,'L_tid':beta}

		self.dtn.compile(loss=losses, loss_weights=lossWeights, optimizer=self.optimizer)

	def train(self, epochs, batch_size):

		y_1 = np.zeros((batch_size,3))
		y_1[:,2] = np.ones(batch_size)	# [0,0,1] for G(x_s)
		y_2 = np.zeros((batch_size,3))
		y_2[:,1] = np.ones(batch_size)	# [0,1,0] for G(x_t)
		y_3 = np.zeros((batch_size,3))
		y_3[:,0] = np.ones(batch_size)	# [1,0,0] for x_t

		y_gang = np.concatenate((y_3,y_3))

		for epoch in range(epochs):
			for batch in tqdm(range(int(faces.shape[0]/batch_size))):

				x_T = load_bitmojis(batch_size)	
				x_S = load_faces(batch_size)

				# Important!! :
				# Normalize bitmoji_imgs: update later
				# Normalize faces: update later

				f_x_S = self.encoder_f.predict(x_S)
				f_x_T = self.encoder_f.predict(x_T)

				g_x_S = self.decoder_g.predict(f_x_S)
				g_x_T = self.decoder_g.predict(f_x_T)

				L_D1, acc_D1 = self.discriminator.train_on_batch(g_x_S, y_1)
				L_D2, acc_D2 = self.discriminator.train_on_batch(g_x_T, y_2)
				L_D3, acc_D3 = self.discriminator.train_on_batch(x_T, y_3)

				L_D = L_D1 + L_D2 + L_D3
				acc_D = (acc_D1 + acc_D2 + acc_D3)/3

				x_dtn = np.concatenate((x_S,x_T))

				y_const = list(zip(np.ones(batch_size),f_x_S))
				y_const.extend(list(zip(np.zeros(batch_size),np.zeros(batch_size))))

				y_tid = list(zip(np.zeros(batch_size),np.zeros(batch_size)))
				y_tid.extend(list(zip(np.ones(batch_size),x_T)))

				L_dtn = self.dtn.train_on_batch(x_dtn, [y_gang, y_const, y_tid])

				print("epoch: "+ str(epoch)+ ", batch_count: "+str(batch) + ", L_D: "+ str(L_D)+ ", L_dtn: "+ str(L_dtn)+ ", accuracy:"+ str(acc_D))


facedet_cascade_path = ''
facenet_model_path = ''
dtn = DTN(facedet_cascade_path, facenet_model_path)