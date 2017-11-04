sup = True

# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    objects.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: scollet <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2017/09/05 22:24:21 by scollet           #+#    #+#              #
#    Updated: 2017/09/05 22:24:22 by scollet          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential

HIDDEN_LAYERS = 2
NEURAL_DENSITY = 32
LEARNING_RATE = 0.001

class Network():
	def __init__(self, d1, d2, d3, out_put):
		self.output_size = out_put
		self.learning_rate = LEARNING_RATE
		self.model = self._build_model(d1, d2, d3, out_put)
		self.target_model = self._build_model(d1, d2, d3, out_put)
	def _build_model(self, d1, d2, d3, out_put):
		model = Sequential()
		model.add(Conv2D(32, (3, 3), input_shape=(d1, d2, d3), data_format='channels_first'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(32, (3, 3), input_shape=(d1, d2, d3), data_format='channels_first'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(64, (3, 3), input_shape=(d1, d2, d3), data_format='channels_first'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(out_put))
		model.add(Activation('relu'))

		model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
		return model
	def pre_gen(self, predict_gen, steps_per_epoch, \
	epochs, validation_data, validation_steps):
		self.model.fit_generator(
			predict_gen,
	        steps_per_epoch=steps_per_epoch,
	        epochs=epochs,
	        validation_data=validation_data,
	        validation_steps=validation_steps)
	def fit_gen(self, train_generator, steps_per_epoch, \
				validation_data, validation_steps):
		self.model.fit_generator(
	        train_generator,
	        steps_per_epoch=steps_per_epoch,
	        validation_data=validation_data,
	        validation_steps=validation_steps)
	def train(self, feed, target):
		self.model.fit(feed, target, epochs=1, verbose=0)
	def predict(self, X):
		self.model.predict(X)
	def load(self, name):
		self.model.load_weights(name)
	def save(self, name):
		self.model.save_weights(name)

class DataGenerator(object):
	'Generates data for Keras'
	def __init__(self, dim_x = 32, dim_y = 32, dim_z = 32, batch_size = 32, shuffle = True):
		'Initialization'
		self.dim_x = dim_x
		self.dim_y = dim_y
		self.dim_z = dim_z
		self.batch_size = batch_size
		self.shuffle = shuffle

	def generate(self, labels, list_IDs):
		'Generates batches of samples'
		# Infinite loop
		while 1:
			# Generate order of exploration of dataset
			indexes = self.__get_exploration_order(list_IDs)

			# Generate batches
			imax = int(len(indexes)/self.batch_size)
			for i in range(imax):
				# Find list of IDs
				list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

				# Generate data
				X, y = self.__data_generation(labels, list_IDs_temp)

				yield X, y

	def __get_exploration_order(self, list_IDs):
		'Generates order of exploration'
		# Find exploration order
		indexes = np.arange(len(list_IDs))
		if self.shuffle == True:
			np.random.shuffle(indexes)
		return indexes

	def __data_generation(self, labels, list_IDs_temp):
		'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, n_channels)
		# Initialization
		X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
		y = np.empty((self.batch_size), dtype = int)

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			#print np.load(ID)
			#print X
			X[0:32, 0:150, 0:150, 0:3] = np.load(ID)
			X_ = np.reshape(X, (X.shape[0], 3, 150, 150))
			#print X.size
			print (X_)
			#np.transpose(X, (32, 150, 150, 3))
			#X =
			#print X

			#print X
			y[i] = labels[ID]

		return X_, sparsify(y)

def sparsify(y):
  'Returns labels in binary NumPy array'
  n_classes = 1# Enter number of classes
  return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                   for i in range(y.shape[0])])
