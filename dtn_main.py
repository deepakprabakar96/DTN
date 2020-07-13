# coding=utf-8
'''

Updates
-------
1. Need to update function load_bitmojis() --> Completed
2. Need to update function load_faces() --> Completed
3. Need to add code for normalizing face and bitmoji images in the function train() --> Completed
4. Added TensorBoard support that needs to be tested
5. Added function to save model and optimizer weights
6. Changed loading of source images to exclude images where faces aren't detected

To-do
-----
1. Mechanism to load model and optimizer weights
2. Test model and weight saving system, TensorBoard callback

??
Change encoded_op_shape
Change facedet_cascade_path, facenet_model_path
'''

from facenet.preprocessing import align_images
from facenet.preprocessing import prewhiten

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras import backend as K
import keras

import tensorflow as tf
import TensorBoard

import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2
import os
from tqdm import tqdm


class DTN:
	def __init__(self, facedet_cascade_path, facenet_model_path, source_path, no_faceslist_path, target_path, batch_save_frequency=100):
		self.img_rows = 160
		self.img_cols = 160
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)

		self.optimizer = Adam(0.0002, 0.5)

		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

		self.cascade_facedet = cv2.CascadeClassifier(facedet_cascade_path)

		self.encoder_f = load_model(facenet_model_path)
		self.encoder_f.name += '_0'
		self.encoder_f.trainable = False

		self.encoder_f2 = load_model(facenet_model_path)
		self.encoder_f2.name += '_1'
		self.encoder_f2.trainable = False

		self.decoder_g = self.build_decoder_g()

		self.discriminator.trainable = False

		self.log_path = "./logs"
		self.save_path = "./model"
		self.batch_save_frequency = batch_save_frequency

		# all class members should be initialized in the init function first
		self.dtn = Model()
		self.build_dtn()

		# source_path/source_image
		no_faces_list = list(np.load(no_faceslist_path))
		self.source_path = source_path
		self.source_images = [image for image in os.listdir(source_path) if image.endswith(".jpg") and
																			image not in no_faces_list]
		self.n_source_images = len(self.source_images)

		# target_path/target_dict.key/target_dict.value
		self.target_path = target_path
		self.target_dict = {}
		target_dirs = os.listdir(target_path)
		for target_dir in target_dirs:
			target_dir_path = os.path.join(self.target_path, target_dir)
			if not os.path.isdir(target_dir_path):
				self.target_dict[target_dir] = [image for image in os.listdir(target_dir_path) if image.endswith(".png")]

		self.train_batchsize = 128

	def build_discriminator(self):

		init = RandomNormal(stddev=0.02)
		model = Sequential()

		model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init, input_shape=self.img_shape))
		model.add(LeakyReLU(alpha=0.2))

		model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
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
		model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
		model.add(LeakyReLU(alpha=0.2))
		# 160x160x128:
		model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
		model.add(LeakyReLU(alpha=0.2))
		# 160x160x3:
		model.add(Conv2D(3, (7, 7), activation='tanh', padding='same', kernel_initializer=init))
		return model

	@staticmethod
	def L_const_wrapper(source):
		def L_const(y_true, y_pred):
			return source*(y_true - y_pred)**2
		return L_const

	@staticmethod
	def L_tid_wrapper(source):
		def L_tid(y_true, y_pred):
			return (1-source)*(y_true - y_pred)**2
		return L_tid

	def build_dtn(self):
		alpha = 100
		beta = 1

		source = Input(shape=(1,))
		inp = Input(shape=self.img_shape)
		encoded_op = self.encoder_f(inp)
		generator_op = self.decoder_g(encoded_op)

		discriminator_op = self.discriminator(generator_op)

		encoded_op2 = self.encoder_f2(generator_op)

		self.dtn = Model(inputs=[inp, source], outputs=[discriminator_op, encoded_op2, generator_op])

		losses = ['categorical_crossentropy', self.L_const_wrapper(source), self.L_tid_wrapper(source)]
		loss_weights = [1, alpha, beta]

		self.dtn.compile(loss=losses, loss_weights=loss_weights, optimizer=self.optimizer)

		print("\n\n" + "*" * 15)
		print("DTN SUMMARY:")
		print(self.dtn.summary())

		plot_model(self.dtn, to_file='./dtn_plot.png', show_shapes=True, show_layer_names=True)

	@staticmethod
	def trim_around_images(image, margin=20):
		h, w, c = image.shape
		trimmed_image = image[int(h * margin / 100):int(h * (100 - margin) / 100),
									int(w * margin / 100):int(w * (100 - margin) / 100), :]
		return trimmed_image

	def load_target(self, batch_size=None):
		if not batch_size:
			batch_size = self.train_batchsize

		key = np.random.choice(self.target_dict.keys())
		subdir_image_paths = [os.path.join(self.target_path, self.target_dict[key], image_name)
												for image_name in np.random.choice(self.target_dict[key], batch_size)]
		batch_images = [cv2.imread(image_path) for image_path in subdir_image_paths]
		trimmed_batch_images = [self.trim_around_images(image) for image in batch_images]
		prewhited_batch_images = [prewhiten(image) for image in trimmed_batch_images]
		batch_as_numpy = np.empty((160, 160, 3, batch_size))
		for i in range(batch_size):
			batch_as_numpy[:, :, :, i] = prewhited_batch_images[i]
		return batch_as_numpy

	def load_source(self, batch_size=None):
		if not batch_size:
			batch_size = self.train_batchsize
		batch_image_paths = [os.path.join(self.source_path, image_name) for image_name in
												np.random.choice(self.source_images, batch_size)]
		batch_images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in batch_image_paths]
		batch_images_aligned = [self.encoder_preprocess(image) for image in batch_images]
		batch_as_numpy = np.empty((160, 160, 3, batch_size))
		for i in range(batch_size):
			batch_as_numpy[:, :, :, i] = batch_images_aligned[i]
		return batch_as_numpy

	@staticmethod
	def write_log(callback, names, logs, batch_no):
		# ------------------------------------------------------ #
		# ----- Check link for TensorBoard implementation ------ #
		# https://github.com/eriklindernoren/Keras-GAN/issues/52 #
		# ------------------------------------------------------ #

		for name, value in zip(names, logs):
			summary = tf.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value
			summary_value.tag = name
			callback.writer.add_summary(summary, batch_no)
			callback.writer.flush()

	def save_model(self, model, model_type, batch_number):
		"""
		:param model: model to be saved
		:param model_type: discriminator/generator
		:param batch_number: batch number (duh!)
		:return: None
		"""
		# # https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state # #

		model_prefix = model_type + "_" + str(batch_number)
		symbolic_weights = getattr(model.optimizer, "weights")
		weight_values = K.batch_get_value(symbolic_weights)
		model_path = os.path.join(self.save_path, model_prefix + ".h5")
		weight_path = os.path.join(self.save_path, model_prefix + "weights.pkl")

		model.save_weights(model_path)
		with open(weight_path, 'wb') as f:
			pickle.dump(weight_values, f)

	def train(self, epochs, batch_size):

		y_1 = np.zeros((batch_size, 3))
		y_1[:, 2] = np.ones(batch_size)	 # [0,0,1] for G(x_s)
		y_2 = np.zeros((batch_size, 3))
		y_2[:, 1] = np.ones(batch_size)	 # [0,1,0] for G(x_t)
		y_3 = np.zeros((batch_size, 3))
		y_3[:, 0] = np.ones(batch_size)	 # [1,0,0] for x_t

		y_gang = np.concatenate((y_3, y_3))

		# --------------------------------------------------------------------- #
		# # Alternate method using keras callback --> callback.on_epoch_end() # #
		# # https://gist.github.com/erenon/91f526302cd8e9d21b73f24c0f9c4bb8   # #
		# --------------------------------------------------------------------- #
		# d_callback = keras.callbacks.TensorBoard(log_dir=self.log_path, histogram_freq=0, batch_size=batch_size,
		# 											write_graph=True, write_grads=True)
		# d_callback.set_model(self.discriminator)
		# g_callback = keras.callbacks.TensorBoard(log_dir=self.log_path, histogram_freq=0, batch_size=batch_size,
		# 											write_graph=True, write_grads=True)
		# g_callback.set_model(self.dtn)
		# --------------------------------------------------------------------- #

		d_callback = TensorBoard(self.log_path)
		d_callback.set_model(self.discriminator)
		g_callback = TensorBoard(self.log_path)
		g_callback.set_model(self.dtn)

		num_batches_per_epoch = int(self.n_source_images/batch_size)
		for epoch in range(epochs):
			for batch in tqdm(range(num_batches_per_epoch)):
				batch_number = epoch * num_batches_per_epoch + batch + 1
				x_T = self.load_target(batch_size)
				x_S = self.load_source(batch_size)

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

				if batch_number % self.batch_save_frequency == 0:
					self.save_model(self.discriminator, "discriminator", batch_number)

				self.write_log(d_callback, ['ld1', 'ld2', 'ld3', 'ld', 'ad1', 'ad2', 'ad3', 'ad'],
									[L_D1, L_D2, L_D3, L_D, acc_D1, acc_D2, acc_D3, acc_D], batch_number)

				x_dtn = np.concatenate((x_S, x_T))

				source = np.concatenate((np.ones(batch_size), np.zeros(batch_size)))

				y_const = np.concatenate((f_x_S, np.zeros_like(f_x_S)))

				y_tid = np.concatenate((np.zeros_like(x_T), x_T))

				L_dtn = self.dtn.train_on_batch([x_dtn, source], [y_gang, y_const, y_tid])

				if batch_number % self.batch_save_frequency == 0:
					self.save_model(self.dtn, "generator", batch_number)

				self.write_log(g_callback, ['lg'], [L_dtn], batch_number)

				print("epoch: " + str(epoch) + ", batch_count: " + str(batch) + ", L_D: " + str(L_D) + ", L_dtn: " +
										str(L_dtn) + ", accuracy:" + str(acc_D))


if __name__ == "__main__":
	facedet_cascade_path = ''
	facenet_model_path = ''
	dtn = DTN(facedet_cascade_path, facenet_model_path)
	dtn.train()