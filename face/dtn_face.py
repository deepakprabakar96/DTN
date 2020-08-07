# coding=utf-8

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
from keras.callbacks import TensorBoard

import pickle
import numpy as np
from skimage.transform import resize
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


class DTN:
	def __init__(self, facedet_cascade_path, facenet_model_path, source_path, source_list_path, target_path,
								output_path,  train_batchsize=16, batch_save_frequency=100, verbose=False, from_ckpt=False, predict=False):
		'''
		Params:

		facedet_cascade_path: haar cascade classifier path
		facenet_model_path:   facenet model path, string
		source_path:		  source dataset folder path, string
		source_list_path:	  path of pickle file containing list of images from source dataset for which haar cascade model detected faces, string
		target_path:		  target dataset folder path, string
		output_path:		  output folder path where generated images will be saved, string
		train_batchsize:	  train batch size, int
		batch_save_frequency: number of batches after which models and generated images will be saved, int
		verbose:			  whether to print log information, bool
		from_ckpt:			  checkpoint model information, data type can be one of the following:
							  int: checkpoint number, picks models with the given checkpoint number
							  tuple: tuple of strings containing paths of weights in order: (d_weights.h5, d_optimizer.pkl, dtn_weights.h5, dtn_optimizer.pkl)
							  bool: picks latest checkpoint from save_path folder if True, trains from scratch if False
		predict:			  whether to use the script as inference script, bool
		'''
		self.verbose = verbose

		self.log_path = "./logs"
		self.save_path = "./model"

		if not os.path.exists(self.log_path):
			os.makedirs(self.log_path)
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)

		self.batch_save_frequency = batch_save_frequency

		# Initialize model and optimizer weight paths
		self.from_ckpt = True
		self.weight_paths = ()
		self.ckpt_number = -1

		# initializing checkpoint information if from_ckpt = True
		self.initialize_ckpt_paths(from_ckpt)

		self.img_rows = 160
		self.img_cols = 160
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)

		self.train_batchsize = train_batchsize
		self.optimizer = Adam(0.0002, 0.5)

		self.discriminator = self.build_discriminator()

		if self.from_ckpt:
			self.discriminator.load_weights(self.weight_paths[0])
			if self.verbose: print("Discriminator model loaded!\n")

		self.discriminator.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

		if not self.from_ckpt:
			if self.verbose: print("Discriminator built and compiled!\n")

		self.cascade_facedet = cv2.CascadeClassifier(facedet_cascade_path)

		if self.verbose: print("Face detection model Loaded!\n")

		self.encoder_f = load_model(facenet_model_path)
		self.encoder_f.name += '_0'
		self.encoder_f.trainable = False

		if self.verbose: print("Encoder_0 loaded!\n")

		self.encoder_f2 = load_model(facenet_model_path)
		self.encoder_f2.name += '_1'
		self.encoder_f2.trainable = False

		if self.verbose: print("Encoder_1 loaded!\n")

		if self.from_ckpt:
			self.decoder_g = load_model(self.weight_paths[2])
			if self.verbose: print("Generator model loaded!\n")
		else:
			self.decoder_g = self.build_decoder_g()
			if self.verbose: print("Generator built!\n")

		self.discriminator.trainable = False

		self.dtn = Model()
		self.build_dtn()

		if self.verbose: print("DTN model built!\n")

		# load source dataset
		self.source_path = source_path

		f = open(source_list_path, 'rb')
		self.source_images = pickle.load(f)
		f.close()
		self.n_source_images = len(self.source_images)

		if self.verbose: print("Source dataset processed!\n")

		# load target dataset
		self.target_path = target_path
		self.target_images = os.listdir(target_path)

		if self.verbose: print("Target dataset processed!\n")

		self.output_path = output_path

		# build network to get inference
		self.predict = predict
		self.pred_model = Model()
		self.build_pred_network()

	def initialize_ckpt_paths(self, from_ckpt):
		all_ckpts = list(set([int(model_name[:-3].split("_")[-1]) for model_name in os.listdir(self.save_path)
								if model_name.endswith(".h5")]))

		if type(from_ckpt) == int:  # Provide batch number to be picked up
			if self.verbose: print("Batch number provided\n")
			ckpt_number = from_ckpt
			self.ckpt_number = ckpt_number
			if ckpt_number in all_ckpts:
				self.from_ckpt = True
				d_weights_path = os.path.join(self.save_path, "discriminator_" + str(ckpt_number) + ".h5")
				d_optimizer_path = os.path.join(self.save_path, "discriminator_" + str(ckpt_number) + "_weights.pkl")
				g_model_path = os.path.join(self.save_path, "generator_" + str(ckpt_number) + ".h5")
				dtn_optimizer_path = os.path.join(self.save_path, "dtn_" + str(ckpt_number) + "_weights.pkl")
				self.weight_paths = (d_weights_path, d_optimizer_path, g_model_path, dtn_optimizer_path)
			else:
				self.from_ckpt = False

		elif isinstance(from_ckpt, tuple):  # Provide paths of model and optimizer weights directly
			if self.verbose: print("Weight paths provided\n")
			self.from_ckpt = True
			self.weight_paths = from_ckpt  # (d_weights.h5, d_optimizer.pkl, dtn_weights.h5, dtn_optimizer.pkl)

		elif from_ckpt==True:  # Pick up most recent checkpoint
			if self.verbose: print("Latest weight paths taken\n")
			if all_ckpts:
				self.from_ckpt = True
				ckpt_number = max(all_ckpts)
				self.ckpt_number = ckpt_number
				d_weights_path = os.path.join(self.save_path, "discriminator_" + str(ckpt_number) + ".h5")
				d_optimizer_path = os.path.join(self.save_path, "discriminator_" + str(ckpt_number) + "_weights.pkl")
				g_model_path = os.path.join(self.save_path, "generator_" + str(ckpt_number) + ".h5")
				dtn_optimizer_path = os.path.join(self.save_path, "dtn_" + str(ckpt_number) + "_weights.pkl")
				self.weight_paths = (d_weights_path, d_optimizer_path, g_model_path, dtn_optimizer_path)
			else:
				self.from_ckpt = False

		else:  # Train from scratch
			if self.verbose: print("No weights chosen; training from scratch\n")
			self.from_ckpt = False

		if self.from_ckpt:
			assert len(self.weight_paths) == 4

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
		encoded_op_shape = 128
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
	def L_custom_wrapper(source):
		def L_custom(y_true, y_pred):
			return source*(y_true - y_pred)**2
		return L_custom

	def build_dtn(self):
		alpha = 16
		beta = 16

		source_const = Input(shape=(1,))
		source_tid = Input(shape=(1,1,1))
		inp = Input(shape=self.img_shape)
		encoded_op = self.encoder_f(inp)
		generator_op = self.decoder_g(encoded_op)

		discriminator_op = self.discriminator(generator_op)

		encoded_op2 = self.encoder_f2(generator_op)

		self.dtn = Model(inputs=[inp, source_const, source_tid], outputs=[discriminator_op, encoded_op2, generator_op])

		losses = ['categorical_crossentropy', self.L_custom_wrapper(source_const), self.L_custom_wrapper(source_tid)]
		loss_weights = [1, alpha, beta]

		self.dtn.compile(loss=losses, loss_weights=loss_weights, optimizer=self.optimizer)

		print("\n\n" + "*" * 15)
		print("DTN SUMMARY:")
		print(self.dtn.summary())

		plot_model(self.dtn, to_file='./dtn_plot.png', show_shapes=True, show_layer_names=True)

	@staticmethod
	def trim_around_images(image, margin=20):
		# Used to trim target dataset around the edges of a face to match source dataset after using haar cascade
		h, w, c = image.shape
		trimmed_image = image[int(h * margin / 100):int(h * (100 - margin) / 100),
									int(w * margin / 100):int(w * (100 - margin) / 100), :]
		return trimmed_image

	def load_target(self, batch_size=None):
		if not batch_size:
			batch_size = self.train_batchsize

		batch_image_paths = [os.path.join(self.target_path, image_name) for image_name in np.random.choice(self.target_images, batch_size)]

		batch_images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in batch_image_paths]
		batch_images_trimmed = [self.trim_around_images(image) for image in batch_images]
		batch_images_resized = [resize(image, (self.img_rows, self.img_cols), mode='reflect')
								for image in batch_images_trimmed]

		batch_images_resized = np.array(batch_images_resized)
		return prewhiten(batch_images_resized)

	def load_source(self, batch_size=None):
		if not batch_size:
			batch_size = self.train_batchsize
		batch_image_paths = [os.path.join(self.source_path, image_name) for image_name in
												np.random.choice(self.source_images, batch_size)]
		batch_images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in batch_image_paths]
		batch_images_aligned = [self.encoder_preprocess(image) for image in batch_images]

		batch_images_aligned = np.array(batch_images_aligned)
		return prewhiten(batch_images_aligned)

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
		model_prefix = model_type + "_" + str(batch_number)

		if model_type == 'discriminator':
			model_path = os.path.join(self.save_path, model_prefix + ".h5")
			model.save_weights(model_path)

		if model_type != 'generator':
			symbolic_weights = getattr(model.optimizer, "weights")
			weight_values = K.batch_get_value(symbolic_weights)
			weight_path = os.path.join(self.save_path, model_prefix + "_weights.pkl")
			with open(weight_path, 'wb') as f:
				pickle.dump(weight_values, f)
		else:
			model_path = os.path.join(self.save_path, model_prefix + ".h5")
			model.save(model_path)

	def train(self, epochs):
		if self.verbose: print("Training Started!\n")
		batch_size = self.train_batchsize
		y_1 = np.zeros((batch_size, 3))
		y_1[:, 2] = np.ones(batch_size)  # [0,0,1] for G(x_s)
		y_2 = np.zeros((batch_size, 3))
		y_2[:, 1] = np.ones(batch_size)  # [0,1,0] for G(x_t)
		y_3 = np.zeros((batch_size, 3))
		y_3[:, 0] = np.ones(batch_size)  # [1,0,0] for x_t

		y_gang = np.concatenate((y_3, y_3))

		# --------------------------------------------------------------------- #
		# # Alternate method using keras callback --> callback.on_epoch_end() # #
		# # https://gist.github.com/erenon/91f526302cd8e9d21b73f24c0f9c4bb8   # #
		# --------------------------------------------------------------------- #
		# d_callback = keras.callbacks.TensorBoard(log_dir=self.log_path, histogram_freq=0, batch_size=batch_size,
		#                                           write_graph=True, write_grads=True)
		# d_callback.set_model(self.discriminator)
		# g_callback = keras.callbacks.TensorBoard(log_dir=self.log_path, histogram_freq=0, batch_size=batch_size,
		#                                           write_graph=True, write_grads=True)
		# g_callback.set_model(self.dtn)
		# --------------------------------------------------------------------- #

		d_callback = TensorBoard(self.log_path)
		d_callback.set_model(self.discriminator)
		g_callback = TensorBoard(self.log_path)
		g_callback.set_model(self.dtn)

		if self.ckpt_number != -1 and self.from_ckpt:
			batch_number = self.ckpt_number
			epochs_from_batch_number = np.round(batch_number * batch_size / self.n_source_images, 3)
		else:
			batch_number = 0
			epochs_from_batch_number = 0

		while epochs_from_batch_number <= epochs:
			batch_number = batch_number + 1
			epochs_from_batch_number = np.round(batch_number * batch_size / self.n_source_images, 3)
			x_T = self.load_target(batch_size)
			x_S = self.load_source(batch_size)

			if self.predict and not self.from_ckpt:
				print('from_ckpt cannot be False')
				break

			if self.predict or batch_number % self.batch_save_frequency == 0:

				pred_x_T = self.pred_model.predict(x_T)
				pred_x_S = self.pred_model.predict(x_S)

				os.mkdir(self.output_path+'/{}_batches'.format(batch_number))

				for i in range(batch_size):
					fig, axs = plt.subplots(2,2)
					axs[0,0].imshow(x_T[i])
					axs[0,1].imshow(pred_x_T[i])
					axs[1,0].imshow(x_S[i])
					axs[1,1].imshow(pred_x_S[i])
					fig.savefig(self.output_path+"/{0}_batches/{1}.png".format(batch_number,i))
					if self.predict:
						plt.show()
				if self.predict:
					break

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

			self.write_log(d_callback, ['D1_LOSS', 'D2_LOSS', 'D3_LOSS', 'D_LOSS', 'D1_ACC', 'D2_ACC', 'D3_ACC', 'D_ACC'],
								[L_D1, L_D2, L_D3, L_D, acc_D1, acc_D2, acc_D3, acc_D], batch_number)

			x_dtn = np.concatenate((x_S, x_T))

			source_const = np.concatenate((np.ones(batch_size), np.zeros(batch_size)))
			source_tid = np.concatenate((np.zeros((batch_size, 1, 1, 1)), np.ones((batch_size, 1, 1, 1))))

			y_const = np.concatenate((f_x_S, np.zeros_like(f_x_S)))

			y_tid = np.concatenate((np.zeros_like(x_T), x_T))

			L_dtn = self.dtn.train_on_batch([x_dtn, source_const, source_tid], [y_gang, y_const, y_tid])

			# set model weights and optimizer states after training one batch:
			if self.from_ckpt and batch_number == 1:
				with open(self.weight_paths[3], 'rb') as f:
					opt_values = pickle.load(f)
				self.dtn.optimizer.set_weights(opt_values)

				with open(self.weight_paths[1], 'rb') as f:
					opt_values = pickle.load(f)
				self.discriminator.optimizer.set_weights(opt_values)

			if batch_number % self.batch_save_frequency == 0:
				self.save_model(self.decoder_g, "generator", batch_number)
				self.save_model(self.dtn, "dtn", batch_number)

			self.write_log(g_callback, ['G_LOSS', 'L_GANG', 'L_CONST', 'L_TID'],
							[L_dtn[0], L_dtn[1], L_dtn[2], L_dtn[3]], batch_number)

			print("batch number: " + str(batch_number) + ", calculated epoch: " + str(epochs_from_batch_number)
								+ ", L_D: " + str(L_D) + ", L_dtn: " + str(L_dtn) + ", accuracy:" + str(acc_D))

		if self.verbose: print("Training completed!\n")

	def build_pred_network(self):
		inp = Input(shape=self.img_shape)
		encoded_op = self.encoder_f(inp)
		generator_op = self.decoder_g(encoded_op)
		self.pred_model = Model(inputs=inp, outputs=generator_op)


if __name__ == "__main__":
	facedet_cascade_path = './facenet/haarcascade_frontalface_alt2.xml'
	facenet_model_path = './facenet/facenet_keras.h5'
	source_path = './img_align_celeba'
	target_path = './bitmoji_data'
	source_list_path = './source_list.pkl'
	output_path = './outputs'

	verbose = True
	dtn = DTN(facedet_cascade_path, facenet_model_path, source_path, source_list_path, target_path, output_path, verbose=verbose, batch_save_frequency=160, from_ckpt=False, train_batchsize=16, predict=False)
	dtn.train(epochs=10)
