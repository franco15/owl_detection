import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import sys
import objects
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from os import listdir
import os
from os.path import isfile, join
import cv2
import PIL
from PIL import Image
import shutil
import csv
#from preprocess import process

train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

# def custom_test_gen():
#     #loading data
#     (X_train, y_train), (X_test, y_test) = train_datagen.flow_from_directory(
#         'data/train/',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')
#
#     #some preprocessing
#     y_train = np_utils.to_categorical(y_train,10)
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     X_train = X_train.astype('float32')
#     X_test = X_test.astype('float32')
#     X_train /= 255
#     X_test /= 255
#     while 1:
#         for i in range(1875):
#             if i%125==0:
#                 print "i = " + str(i)
#             yield X_train[i*32:(i+1)*32], y_train[i*32:(i+1)*32]

# def data_gen(top_dim, bot_dim):
#     """
#     Generator to yield batches of two inputs (per sample) with shapes top_dim and
#     bot_dim along with their labels.
#     """
#     batch_size = 264
#     while True:
#         top_batch = []
#         bot_batch = []
#         batch_labels = []
#         for i in range(batch_size):
#             # Create random arrays
#             rand_pix = np.random.randint(100, 256)
#             top_img = np.full(top_dim, rand_pix)
#             bot_img = np.full(bot_dim, rand_pix)
#
#             # Set a label
#             label = np.random.choice([0, 1])
#             batch_labels.append(label)
#
#             # Pack each input image separately
#             top_batch.append(top_img)
#             bot_batch.append(bot_img)
#
#         yield [np.array(top_batch), np.array(bot_batch)], np.array(batch_labels)

# def custom_valid_gen():
#     #loading data
#     (X_train, y_train), (X_test, y_test) = train_datagen.flow_from_directory(
#         'data/train/',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')
#
#     #some preprocessing
#     y_train = np_utils.to_categorical(y_train,10)
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     X_train = X_train.astype('float32')
#     X_test = X_test.astype('float32')
#     X_train /= 255
#     X_test /= 255
#     while 1:
#         for i in range(1875):
#             if i%125==0:
#                 print "i = " + str(i)
#             yield X_train[i*32:(i+1)*32], y_train[i*32:(i+1)*32]

def train(model):
	batch_size = 32

	#test_datagen = ImageDataGenerator(rescale=1./255)

	params = {'dim_x': 150,
          'dim_y': 150,
          'dim_z': 3,
          'batch_size': batch_size,
          'shuffle': True}

	#thefile = open('test.txt', 'w')
	# Datasets
	if os.path.exists('data/train/npy'):
		shutil.rmtree('data/train/npy')
		os.makedirs('data/train/npy')
	else:
		os.makedirs('data/train/npy')
	if os.path.exists('data/validation/npy'):
		shutil.rmtree('data/validation/npy')
		os.makedirs('data/validation/npy')
	else:
		os.makedirs('data/validation/npy')

	print ("read training")
	basewidth = 150

	for each in range(0, len(listdir('data/train/images'))):
		Path = 'data/train/images/' + listdir('data/train/images')[each]
		img = Image.open(Path)
		#wpercent = (basewidth / float(img.size[0]))
		#hsize = 150
		img = img.resize((150, 150), PIL.Image.BICUBIC)
		# img.save(Path)
		this = cv2.imread(Path)
		#print this
		np.save('data/train/npy/' + 'TRAIN' + str(each) + '.npy', this)
		#i += 1

	print ("read valid")
	for each in range(0, len(listdir('data/validation/images'))):
		Path = 'data/validation/images/' + listdir('data/validation/images')[each]
		img = Image.open(Path)
		#wpercent = (basewidth / float(img.size[0]))
		#hsize = int((float(img.size[1]) * float(wpercent)))
		img = img.resize((150, 150), PIL.Image.BICUBIC)
		# img.save(Path)
		this = cv2.imread(Path)
		#print this
		np.save('data/validation/npy/' + 'TRAIN' + str(each) + '.npy', this)

	#os.Exit(1)

	partition = {'train':[], 'validation':[]}
	labels = {'owls': []}

	print ("partition train")
	for f in listdir('data/train/npy'):
		partition['train'].append('data/train/npy/' + f)
		labels['data/train/npy/' + f] = 1

	print ("partition valid")
	for f in listdir('data/validation/npy'):
		partition['validation'].append('data/validation/npy/' + f)
		labels['data/validation/npy/' + f] = 1

	# Generators
	#print partition['train']#, partition['validation']
	training_generator = objects.DataGenerator(**params).generate(labels, partition['train'])
	validation_generator = objects.DataGenerator(**params).generate(labels, partition['validation'])
	# train_generator = train_datagen.flow_from_directory(
    #     'data/train/',
    #     target_size=(150, 150),
    #     batch_size=batch_size,
    #     class_mode='binary')

	# validation_generator = test_datagen.flow_from_directory(
    #     'data/validation/',
    #     target_size=(150, 150),
    #     batch_size=batch_size,
    #     class_mode='binary')

	model.fit_gen(
        train_generator = training_generator,
    	steps_per_epoch = len(partition['train'])//batch_size,
        validation_data = validation_generator,
        validation_steps = len(partition['validation'])//batch_size)
	model.save('trained.h5')

def predict(model, img):
	batch_size = 32


# 44 = <
# 46 = >

def manual_classification(train, valid):
	if os.path.exists('train_manual_classification.csv'):
		os.remove('train_manual_classification.csv')
	with open('train_manual_classification.csv', 'w') as train_csv:
		wr = csv.writer(train_csv, delimiter='|')
		for i in range(0, len(listdir(train))):
			t_path = train + '/' + listdir(train)[i]
			img = cv2.imread(t_path)
			resize = cv2.resize(img, (900, 600))
			cv2.imshow('owl or not owl?', resize)
			while True:
				key = cv2.waitKey(0)
				if key is 27:
					break
				elif key is 48:
					break
				elif key is 49:
					break
			if key is 27: # escape key
				break
			elif key is 49: # 1
				wr.writerow((t_path, '1'))
			elif key is 48:
				wr.writerow((t_path, '0'))
			cv2.destroyWindow('owl or not owl?')

		print ('\n\ndone with train_manual_classification, valid_manual_classification about to start\n\n')

	if os.path.exists('valid_manual_classification.csv'):
		os.remove('valid_manual_classification.csv')
	with open('valid_manual_classification.csv', 'w') as valid_csv:
		wr = csv.writer(valid_csv, delimiter='|')
		for i in range(0, len(listdir(valid))):
			t_path = valid + '/' + listdir(valid)[i]
			img = cv2.imread(t_path)
			resize = cv2.resize(img, (900, 600))
			cv2.imshow('owl or not owl?', resize)
			key = cv2.waitKey(0)
			if key is 27: # escape key
				break
			elif key is 49: # 1
				wr.writerow((t_path, '1'))
			elif key is 48:
				wr.writerow((t_path, '0'))
			cv2.destroyWindow('owl or not owl?')

	cv2.destroyAllWindows()

	return train_csv, valid_csv

def run(args):
	model = objects.Network(3, 150, 150, 1)
	if args[1] == '--train':
		train(model)
	elif args[1] == '--predict':
		predict(model, [args[2]])
	elif args[1] == '--class':
		# tmc: manual_classification of images in the train directory
		# cmc: manual_classification of images in the validation directory
		tmc, cmc = manual_classification(args[2], args[3])

if __name__ == "__main__":
	run(sys.argv)
