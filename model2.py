''' 
An attempt to implement the model described by researchers at NVIDIA 
for brain tumor segmentation, described in:
	- Myronenko, Andriy. "3D MRI brain tumor segmentation using autoencoder regularization." 
		International MICCAI Brainlesion Workshop. Springer, Cham, 2018.
'''
from keras.losses import mse
from keras.layers import Conv3D, Reshape, Dropout, SpatialDropout3D, Input, Flatten, Activation, Add, UpSampling3D, Lambda, Dense
from keras.optimizers import adam
from keras.models import Model, Sequential
import keras.backend as K


def sampling(args):
	# keras-team/keras/blob/master/examples/variational_autoencoder.py
    z_mean, z_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_var) * epsilon

def dice_score(y_true, y_pred):
    overlap = K.sum(K.abs(y_true * y_pred), axis=[-3,-2,-1])
    dn = K.sum(K.square(y_true) + K.square(y_pred), axis=[-3,-2,-1]) + 1e-8
    return K.mean(2 * overlap / dn, axis=[0,1])

def gt_loss_wrapper(e=1e-8):	
# 	Wrapper allows more information to be passed
	def gt_loss(y_true, y_pred):
        return dice_score(y_true, y_pred)    
    return gt_loss

def vae_loss_wrapper(input_shape, z_mean, z_var, L2w=0.1, KLw=0.1):
# 	Wrapper allows more information to be passed
	def vae_loss(y_true, y_pred):
		(c, H, W, D) = input_shape
		num_dims = c * H * W * D
		L2l = K.mean(K.square(y_true - y_pred), axis=(1, 2, 3, 4))
		KLl = (1 / num_dims) * K.sum(K.exp(z_var) + K.square(z_mean) - 1 - z_var, axis=-1)
		return L2w * KLw * L2l * KLl
	return vae_loss

def block_a(layers, num_filters, format='channels_first'):
	base = Conv3D(filters=filters, kernel_size=(1, 1, 1), strides=1,
        data_format=format)(layers)
	x = Sequential()
	x.add(Activation('relu'))
	x.add(Conv3D(filters=filters, kernel_size=(3, 3, 3), strides=1,
		padding='same', data_format=format))
	x.add(Activation('relu'))
	x.add(Conv3D(filters=filters, kernel_size=(3, 3, 3), strides=1,
		padding='same', data_format=format))
	output = Add()([x, base])
	return output

def build_model(input_shape=(4, 64, 64, 64), out_channels=3, L2w=0.1, KLw=0.1, exp=1e-8):
	(channels, H, W, D) = input_shape

	'''Input'''    
	base = Input(input_shape)    
	x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=1,
		padding='same', data_format='channels_first')(base)
    x = SpatialDropout3D(0.2, data_format='channels_first')(x)
    x_1 = block_a(x, 32, name='x1')
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=2,
        padding='same', data_format='channels_first')(x_1)
    x = block_a(x, 64)
    x_2 = block_a(x, 64)
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=2,
        padding='same', data_format='channels_first')(x_2)
    x = block_a(x, 128)
    x_3 = block_a(x, 128)
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=2,
        padding='same', data_format='channels_first')(x_3)
    x = block_a(x, 256)
    x = block_a(x, 256)
    x = block_a(x, 256)
    x4 = block_a(x, 256)

    '''Decoder'''
    x = Conv3D(filters=128, kernel_size=(1, 1, 1), strides=1, data_format='channels_first')(x4)
    x = UpSampling3D(size=2, data_format='channels_first')(x)
    x = Add()([x, x_3])
    x = block_a(x, 128)
    x = Conv3D(filters=64, kernel_size=(1, 1, 1), strides=1, data_format='channels_first')(x)
    x = UpSampling3D(size=2, data_format='channels_first')(x)
    x = Add()([x, x_2])
    x = block_a(x, 64)
    x = Conv3D(filters=32, kernel_size=(1, 1, 1),strides=1, data_format='channels_first')(x)
    x = UpSampling3D(size=2, data_format='channels_first')(x)
    x = Add()([x, x_1])
    x = block_a(x, 32)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=1, padding='same', data_format='channels_first')(x)

    '''Output'''
    ground_truth_out = Conv3D(filters=out_channels, kernel_size=(1, 1, 1), strides=1,
        data_format='channels_first', activation='sigmoid')(x)

    '''VAE'''
    x = Activation('relu')(x)
    x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=2, padding='same', data_format='channels_first')(x)
    z_mean = Dense(128)(x)
    z_var = Dense(128)(x)
    x = Lambda(sampling)([z_mean, z_var])
    x = Dense((channels//4) * (H//16) * (W//16) * (D//16))(x)
    x = Activation('relu')(x)
    x = Reshape(((channels//4), (H//16), (W//16), (D//16)))(x)
    x = Conv3D(filters=256, kernel_size=(1, 1, 1), strides=1, data_format='channels_first')(x)
    x = UpSampling3D(size=2, data_format='channels_first')(x)
    x = Conv3D( filters=128, kernel_size=(1, 1, 1), strides=1, data_format='channels_first')(x)
    x = UpSampling3D(size=2, data_format='channels_first')(x)
    x = block_a(x, 128)
    x = Conv3D(filters=64, kernel_size=(1, 1, 1), strides=1, data_format='channels_first')(x)
    x = UpSampling3D(size=2, data_format='channels_first')(x)
    x = block_a(x, 64)
    x = Conv3D(filters=32, kernel_size=(1, 1, 1), strides=1, data_format='channels_first')(x)
    x = UpSampling3D(size=2, data_format='channels_first')(x)
    x = block_a(x, 32)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=1, padding='same', data_format='channels_first')(x)

    '''Output'''
    VAE_out = Conv3D(filters=4, kernel_size=(1, 1, 1), strides=1, data_format='channels_first')(x) 

    # Build and Compile the model
    model = Model(inp, outputs=[ground_truth_out, VAE_out])  # Create the model
    model.compile(adam(lr=1e-4), [gt_loss(exp), vae_loss(input_shape, z_mean, z_var, L2w=L2w, KLw=KLw)], metrics=[dice_score])
    return model
