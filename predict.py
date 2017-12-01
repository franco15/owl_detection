import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import sys


def create_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
	model.add(MaxPooling2D(2, 2))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(2, 2))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))

	model.add(Dense(2, activation='softmax'))

	# model.summary()

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

def predict(args):
	img = cv2.imread(args[1])
	img = cv2.resize(img, (150, 150))
	model = create_model()
	model.load_weights('buhos.h5')
	arr = np.array(img).reshape((150, 150, 3))
	arr = np.expand_dims(arr, axis=0)
	prediction = model.predict(arr)[0]
	print (prediction)
	# bestclass = ''
	# bestconf = -1
	# for i in [0,1]:
	# 	if prediction[i] > bestconf:
	# 		bestclass = str(i)
	# 		bestconf = prediction[i]
	# print ('bestclass: ', bestclass)
	# print ('bestconf: ', bestconf)

if __name__ == '__main__':
	args = sys.argv
	predict(args)
