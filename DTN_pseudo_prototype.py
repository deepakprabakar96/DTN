


def encoder_f(image):
	model  = Sequential()
	model add Dense()
	model add Conv2D()
	model add Conv2D()
	model add Flatten()
	model add Dense()

	# doubtful
	# model compile loss MSE: Lconst
	return model

def decoder_g(f):
	model = Sequential()
	model add Conv2DTranspose(f)
	model add Conv2DTranspose()

	return model


def discriminator(image):
	model = Sequential()
	model add Conv2D(image)
	model add Conv2D()
	model add Flatten()
	model add Dense(3, softmax)
	model compile loss cross_entropy: Ld
	return model


def DTN(D, f, g, input_img_shape):
	D.trainable = False
	f.trainable = False


	dtn_input = Input( input_img_shape)
	f_output = f(dtn_input)
	g_output = g(f_output)
	dtn_output = D(g_output)

	dtn = model ( input: dtn_input, output: dtn_output)

	dtn compile cross_entropy: L_GANG
	

def train_DTN(epochs, batch_size):
	
	image_shape

	Generator G:
	f = encoder_f()
	g = decoder_g()
	D = discriminator()

	dtn = DTN(D, f, g, img_shape)

	Iterate epoch, batches:

		orig_img = load_org()
		x_S = load_S()
		x_T = load_T()

		D.trainable = true
		
		'''
		target vector would be :
			[1,0,0] for orig_img,
			[0,1,0] for x_T,
			[0,0,1] for x_S
		'''

		y_dis = shape(3, 3*batch_size)

		y_dis[:batch_size]               = [1,0,0]*batch_size/3
		y_dis[batch_size:batch_size*2]   = [0,1,0]*batch_size/3
		y_dis[batch_size*2:]             = [0,0,1]*batch_size/3

		Loss_D3, D3_acc = D train_on_batch (orig_img, y_dis[:batch_size])
		Loss_D2, D2_acc = D train_on_batch (x_T, y_dis[batch_size:batch_size*2])
		Loss_D1, D1_acc = D train_on_batch (x_S, y_dis[batch_size*2:])
		
		Loss_D = Loss_D1 + Loss_D2 + Loss_D3


		y_dtn = shape(2, 2*batch_size)
		y_dtn = [1,0,0]*batch_size*2

		D.trainable = False
		f.trainable = False

		dtn_train_input = concatenate (x_S,x_T)
		L_GANG = dtn train_on_batch (dtn_train_input, y_dtn)
		

		L_const = MSE ( )?

		L_TID = MSE ( )?