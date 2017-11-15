import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model(features, labels, mode):
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
	pool2_flat = tf.reshape(pool2, [-1, (75/2) * (75/2) * 64])

	dense = tf.layers.dense(
		inputs=pool2_flat, units=1024, activation=tf.nn.relu)

	# the rate argument specifies the number of elements that will be randomly
	# 	dropped out during training
	droput = tf.layers.droput(
		inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

	# 2 units, one for each target class aka owl, not owl
	logits = tf.layers.dense(inputs=droput, units=2)

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


if __name__ == "__main__":
	tf.app.run()
