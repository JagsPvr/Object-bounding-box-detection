import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras import regularizers
from hyperparams import Hyperparams
from data_loader import Train_data_loader, Val_data_loader
from model import Model
import logging

#logger configuration
FORMAT = "[%(filename)s: %(lineno)3s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

H = Hyperparams()

train_batch_generator = Train_data_loader(H.train_batch_size, H.num_train)
val_batch_generator = Val_data_loader(H.val_batch_size, H.num_train)
logger.info("Generators instantiated")

model = Model().get_model()
logger.info("Model loaded")
adam = tf.keras.optimizers.Adam(lr=H.learning_rate)
model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mean_squared_error'])
model.load_weights("saved_weights/model_epoch_1.h5")
logger.info("Model compiled")

logger.info("Calling fit_generator")
# model.fit_generator(generator=train_batch_generator,
# 	epochs=H.num_epochs,
# 	verbose=2,
# 	steps_per_epoch=(H.num_train//H.train_batch_size),
# 	validation_data=val_batch_generator,
# 	validation_steps=(H.num_val//H.val_batch_size),
# 	shuffle=True,
# 	max_queue_size=32)
# model.evaluate(x_test, y_test)


# from keras.callbacks import Callback
# class LossHistory(Callback) :
# 	'''Callbakc function for collecting loss history of model per batch'''
# 	def on_train_begin(self, log={}) :
# 		self.losses = []
# 	def on_batch_end(self,batch,logs=[]):
# 		self.losses.append(logs.get('loss'))
import numpy as np
LossHistory = []
np.array(LossHistory)

for epoch in range(H.num_epochs) :
	for batch_idx in range(H.num_train // H.train_batch_size) :
		img_batch, lebels_batch = train_batch_generator[batch_idx]
		loss = model.train_on_batch(img_batch,lebels_batch)
		logger.info("Epoch : {}, step : {}, Loss : {}".format(epoch,batch_idx,loss))
		LossHistory.append(loss)
	val_loss = model.evaluate_generator(generator = val_batch_generator.__iter__(), steps = H.num_val//H.val_batch_size,verbose=2, max_queue_size=3)
	logger.info("validation - Epoch : {}, val_loss : {}".format(epoch,val_loss))
	# logger.info("Validation - Epoch : {}, Val_Loss : {}".format(epoch, val_loss))
	model.save_weights("saved_weights/model_epoch_2.h5")
	logger.info("Model weights - model_epoch_2 saved")

import matplotlib.pyplot as plt
plt.figure()
plt.plot(LossHistory)
plt.show()
# plt.plot(LossHistory)

# print(type(val_batch_generator))
# print(dir(val_batch_generator))
# print(val_batch_generator)