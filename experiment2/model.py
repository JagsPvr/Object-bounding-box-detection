from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LeakyReLU
from keras import regularizers
from hyperparams import Hyperparams
H = Hyperparams()

class Model(object):

	def __init__(self):
		return None

	def get_model(self):
		#create model
		model = Sequential()
		LR = LeakyReLU(H.alpha)
		#add model layers
		model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(192,256,3)))
		model.add(Conv2D(64, kernel_size=5, activation='relu'))		
		model.add(MaxPooling2D(pool_size = (2,2),strides=(1,1)))
		model.add(Conv2D(64, kernel_size=7, activation='relu',strides=(2,2)))
		model.add(MaxPooling2D(pool_size = (2,2)))
		model.add(Conv2D(128, kernel_size=9, activation='relu',strides=(2,2)))
		model.add(MaxPooling2D(pool_size = (2,2)))
		model.add(Conv2D(256, kernel_size=7, activation='relu',strides=(3,3)))
		model.add(Flatten())
		model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
		model.add(Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
		model.add(Dense(4, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
		# model.add(Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
		# model.add(Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
		# model.add(Dense(4, activation='sigmoid',kernel_regularizer=regularizers.l2(0.001)))
		return model