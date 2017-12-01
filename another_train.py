import os
import csv
import cv2
import sys
import shutil
import PIL
import numpy as np
import tensorflow as tf
from PIL import Image

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model(features, labels, mode):

	# print ('labels in cnn model')
	# print (labels, '\n\n')
	# model funtion for cnn_model
	# input layer [batch_size, img_width, img_height, channels]
	# note the -1 for batch_sizewhich specifies that this dimension should be
	# 	dynamically computed based on the number of input values in features["x"],
	# 	holding the size of all other dimensions constant.
	input_layer = tf.reshape(features["x"],[-1, 150, 150, 3])

	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=5,
		padding="same",
		activation=tf.nn.relu)

	# pool_size: size of max pooling filter
	# strides: set as 2, indicate that the subregions extracted by the filter
	# should be separated by 2 pixels in both the width and height dimensions
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=5,
		padding="same",
		activation=tf.nn.relu)

	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

	# -1 means that batch_size will be dynamically calculated based on the number
	# 	of examples in our input data
	# each example has 75/2 (pool2 width) * 75/2 (pool2 height) * 64 (pool2 channels)
	pool2_flat = tf.reshape(pool2, [-1, 37 * 37 * 64])

	dense = tf.layers.dense(
		inputs=pool2_flat, units=1024, activation=tf.nn.relu)

	# the rate argument specifies the number of elements that will be randomly
	# 	dropped out during training
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

	# 2 units, one for each target class aka owl, not owl
	logits = tf.layers.dense(inputs=dropout, units=2)

	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
	loss = tf.losses.softmax_cross_entropy(
		onehot_labels=onehot_labels, logits=logits)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss, global_step=tf.train.get_global_step())
		return (tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op))

	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
	}
	return (tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops))


def resize_images(av):
	if os.path.exists('data/resized_images'):
		shutil.rmtree('data/resized_images')
		# os.makedirs('data/resized_images')
		os.makedirs('data/resized_images/train/owls')
		os.makedirs('data/resized_images/train/not_owls')
		os.makedirs('data/resized_images/validation/owls')
		os.makedirs('data/resized_images/validation/not_owls')
	else:
		# os.makedirs('data/resized_images')
		os.makedirs('data/resized_images/train/owls')
		os.makedirs('data/resized_images/train/not_owls')
		os.makedirs('data/resized_images/validation/owls')
		os.makedirs('data/resized_images/validation/not_owls')

	for i in range(0, len(os.listdir(av[2] + '/train/owls'))):
		name = os.listdir(av[2] + '/train/owls')[i]
		path = av[2] + '/train/owls/' + name
		img = Image.open(path)
		resized = img.resize((150, 150), PIL.Image.BICUBIC)
		resized.save('data/resized_images/train/owls/' + name)

	for i in range(0, len(os.listdir(av[2] + '/train/not_owls'))):
		name = os.listdir(av[2] + '/train/not_owls')[i]
		path = av[2] + '/train/not_owls/' + name
		img = Image.open(path)
		resized = img.resize((150, 150), PIL.Image.BICUBIC)
		resized.save('data/resized_images/train/not_owls/' + name)

	for i in range(0, len(os.listdir(av[2] + '/validation/not_owls'))):
		name = os.listdir(av[2] + '/validation/not_owls')[i]
		path = av[2] + '/validation/not_owls/' + name
		img = Image.open(path)
		resized = img.resize((150, 150), PIL.Image.BICUBIC)
		resized.save('data/resized_images/validation/not_owls/' + name)

	for i in range(0, len(os.listdir(av[2] + '/validation/owls'))):
		name = os.listdir(av[2] + '/validation/owls')[i]
		path = av[2] + '/validation/owls/' + name
		img = Image.open(path)
		resized = img.resize((150, 150), PIL.Image.BICUBIC)
		resized.save('data/resized_images/validation/owls/' + name)


def get_data():
	train_dir = 'data/resized_images/train/'
	validation_dir = 'data/resized_images/validation/'

	train_tmp = np.empty((0, 150*150*3), dtype=np.float32)
	validation_tmp = np.empty((0, 150*150*3), dtype=np.float32)
	# tmp = np.empty((1, 150, 150 ,3), dtype=np.int32)
	# print (train_tmp.shape)

	# get train data
	for file_name in os.listdir(train_dir + 'not_owls'):
	# for i in range(0, len(os.listdir(train_dir + 'not_owls'))):
		# file_name = os.listdir(train_dir + 'not_owls')[i]
		path = train_dir + 'not_owls/' + file_name
		# img = Image.open(path)
		img = cv2.imread(path)
		img_arr = np.array(img, dtype=np.float32)
		# print (img_arr)
		train_tmp = np.append(train_tmp, img_arr)
		# print ('tmp')
		# print (tmp.shape)
		# print (i)
		# if i > 2:
			# break
		# train_data = np.reshape(tmp, (-1, 150, 150, 3))
		train_data = np.reshape(train_tmp, (-1, 150*150*3))
		# print ('train_data')
		# print (train_data)
	print (train_data.shape)

	for file_name in os.listdir(train_dir + 'owls'):
		path = train_dir + 'owls/' + file_name
		img = cv2.imread(path)
		img_arr = np.array(img, dtype=np.float32)
		train_tmp = np.append(train_tmp, img_arr)
		train_data = np.reshape(train_tmp, (-1, 150*150*3))
	print (train_data.shape)

	# get validation data

	for file_name in os.listdir(validation_dir + 'not_owls'):
		path = validation_dir + 'not_owls/' + file_name
		img = cv2.imread(path)
		img_arr = np.array(img, dtype=np.float32)
		validation_tmp = np.append(validation_tmp, img_arr)
		validation_data = np.reshape(validation_tmp, (-1, 150*150*3))
	print (validation_data.shape)

	for file_name in os.listdir(validation_dir + 'owls'):
		path = validation_dir + 'owls/' + file_name
		img = cv2.imread(path)
		img_arr = np.array(img, dtype=np.float32)
		validation_tmp = np.append(validation_tmp, img_arr)
		validation_data = np.reshape(validation_tmp, (-1, 150*150*3))
	print (validation_data.shape)
	# return train_data, validation_data
	np.save('data/train_data.npy', train_data)
	np.save('data/validation_data.npy', validation_data)


def get_labels():
	train_dir = 'data/resized_images/train/'
	validation_dir = 'data/resized_images/validation/'
	#
	# owl = np.array([1, 0])
	# not_owl = np.array([0, 1])
	i = 0;
	train_csv = 'data/resized_images/train.csv'
	validation_csv = 'data/resized_images/validation.csv'
	with open(train_csv, 'w') as myfile:
		wr = csv.writer(myfile, delimiter='|')
		for file_name in os.listdir(train_dir + 'owls'):
			wr.writerow('1')
			# i += 1
			# if i > 2:
			# 	break
		i = 0
		for file_name in os.listdir(train_dir + 'not_owls'):
			wr.writerow('0')
			# i += 1
			# if i > 2:
			# 	break
	with open(validation_csv, 'w') as myfile:
		wr = csv.writer(myfile, delimiter='|')
		i = 0
		for file_name in os.listdir(validation_dir + 'owls'):
			wr.writerow('1')
			# i += 1
			# if i > 2:
			# 	break
		i = 0
		for file_name in os.listdir(validation_dir + 'not_owls'):
			wr.writerow('0')
			# i += 1
			# if i > 2:
			# 	break
	train_labels = np.genfromtxt(train_csv, delimiter='|', dtype=np.int32)
	validation_labels = np.genfromtxt(validation_csv, delimiter='|', dtype=np.int32)

	print ('train_labels')
	print (train_labels.shape)
	print ('validation_labels')
	print (validation_labels.shape)

	np.save('data/train_labels.npy', train_labels)
	np.save('data/validation_labels.npy', validation_labels)
	# return train_labels, validation_labels


def main(av):
	if len(av) > 2 and av[1] == '--resize':
		resize_images(av)
	elif len(av) > 1 and av[1] == '--resize':
		print ('missing/too many arguments')
	elif len(av) == 2 and av[1] == '--labels':
		get_labels()
	elif len(av) == 2 and av[1] == '--data':
		get_data()
	# train_data = np.array([])
	# img = Image.open('data/resized_images/train/owls/owl_1.jpg')
	# img2 = cv2.imread('data/resized_images/train/owls/owl_2.jpg')
	# arr = np.array(img)
	# arr2 = np.array(img2)
	# train_data = np.append(arr, arr2, axis=0)
	# print ('arr')
	# print (arr.shape)
	# print ('arr2')
	# print (arr2.shape)
	# print ('train_data')
	# print (train_data.shape)
	# print ('\n\n\n\n')
	# img3 = cv2.imread('data/resized_images/train/owls/owl_3.jpg')
	# arr3 = np.array(img3)
	# train_data = np.append(train_data, arr3)
	# print ('appended another image')
	# print ('arr3')
	# print (arr3)
	# print ('train_data')
	# print (train_data)

	# train_data, validation_data = get_data()
	train_data = np.load('data/train_data.npy')
	validation_data = np.load('data/validation_data.npy')
	print ('train_data')
	print (train_data.dtype)
	print ('validation_data')
	print (validation_data.dtype)
	# train_labels, validation_labels = get_labels()
	train_labels = np.load('data/train_labels.npy')
	validation_labels = np.load('data/validation_labels.npy')
	# print ('train_labels')
	# print (train_labels)
	# print ('validation_labels')
	# print (validation_labels)

	# Create the estimator
	owl_classifier = tf.estimator.Estimator(
		model_fn=cnn_model, model_dir='owl_convnet_model')

	# set up logging for predictions
	tensors_to_log = {'probabilities': 'softmax_tensor'}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=50)

	# train model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={'x': train_data},
		y=train_labels,
		batch_size=32,
		num_epochs=None,
		shuffle=True)
	owl_classifier.train(
		input_fn=train_input_fn,
		steps=20000,
		hooks=[logging_hook])

	# evaluate the model and print results
	validation_input_fn = tf.estimator.inputs.numpy.fn(
		x={'x': validation_data},
		y=validation_labels,
		num_epochs=1,
		shuffle=False)
	validation_results = owl_classifier.evaluate(input_fn=validation_input_fn)
	print (validation_results)


if __name__ == "__main__":
	tf.app.run()
	# main(sys.argv)
