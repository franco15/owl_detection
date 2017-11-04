sup

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

HIDDEN_LAYERS = 2
NEURAL_DENSITY = 32
LEARNING_RATE = 0.001

class Network():
	def __init__(self, d1, d2, d3, out_put):
		self.output_size = out_put
		self.learning_rate = LEARNING_RATE
		self.model = self._build_model()
		self.target_model = self._build_model()
		self.update_target_model()
	def _huber_loss(self, target, prediction):
		error = prediction - target
		return K.mean(K.sqrt(1 + K.square(error)) - 1, axis = -1)
	def _build_model(self):
		model = Sequential()
		model.add(Conv2D(32, (3, 3), input_shape=(d1, d2, d3)))
		model.add(Activation('selu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(32, (3, 3)))
		model.add(Activation('selu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('selu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64))
		model.add(Activation('selu'))
		model.add(Dropout(0.5))
		model.add(Dense(out_put))
		model.add(Activation('elu'))

		model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
		return model
	def train(self, feed, target):
		self.model.fit(feed, target, epochs=1, verbose=0)
	def predict(self, X):
		self.model.predict(X)
	def load(self, name):
		self.model.load_weights(name)
	def save(self, name):
		self.model.save_weights(name)
