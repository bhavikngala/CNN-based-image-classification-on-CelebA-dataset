from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from helpers import celebHelper as zeenat
from helpers import image_helper as davinci
from helpers import fileHelper as owl

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

imageRes = [50, 50]
imageResInput = [-1, 50, 50, 1]
pool2outputSize = 12 * 12 * 64


def readImages():
	imageDirectory = './../data/img_align_celeba/'

	images = davinci.batchReadAndResizeImages(imageDirectory, imageRes, 'bilinear', '.jpg')
	images = np.array(images)

	davinci.plotImage(images[0,:], imageRes)

	owl.writeNumpyArrayToFile('./../data/numpyarray/', 'images150.npy', images)

def cnnModel(features, labels, mode):
	'''Model function for CNN.'''
	# input layer
	input_layer = tf.reshape(features['x'], imageResInput)

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding='same',
		activation=tf.nn.relu)
	print('convolution 1')

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(
		inputs=conv1,
		pool_size=[2, 2],
		strides=2)
	print('pooling 1')

	# Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5,5],
		padding='same',
		activation=tf.nn.relu)
	print('convolution 2')

	pool2 = tf.layers.max_pooling2d(
		inputs=conv2,
		pool_size=[2, 2],
		strides=2)
	print('pooling 2')

	# Dense Layers
	pool2_flat = tf.reshape(pool2, [-1, pool2outputSize])
	print('pooling 2 flat')

	dense = tf.layers.dense(
		inputs=pool2_flat,
		units=1024,
		activation=tf.nn.relu)
	print('dense')

	dropout = tf.layers.dropout(
		inputs=dense,
		rate=0.4,
		training=mode == tf.estimator.ModeKeys.TRAIN)
	print('dropout')

	# Logits Layer
	logits = tf.layers.dense(
		inputs=dropout,
		units=2)
	print('logits')

	predictions = {
		'classes' : tf.argmax(input=logits, axis=1),
		'probabilities': tf.nn.softmax(logits, name="softmax_tensor")
	}
	print('predictions')

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
	loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
		logits=logits)
	print('loss')

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		print('train')
		return tf.estimator.EstimatorSpec(mode=mode,
			loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels, predictions=predictions["classes"])}
	print('evaluation')
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
	imageNameAndLabelsFile = './../data/list_attr_celeba.txt'
	[imageNames, imageLabels] = zeenat.readNamesImageLabels(
		imageNameAndLabelsFile, [15], 2)

	imageLabels = np.asarray(imageLabels, dtype=np.int32)
	imageLabels[imageLabels<0] = 0
	imageLabels = imageLabels.flatten()

	images = owl.readNumpyArrayFromFile(
		'./../data/numpyarray/images50.npy')
	images = images.astype('float32')
	N = images.shape[0]

	[images, imageLabels, vali_images, vali_labels, test_images, test_labels] = \
		zeenat.separateDataSets(images, imageLabels)

	# davinci.plotImage(images[0,:], imageRes)

	# readImages()

	# create the classifier
	glassesClassifier = tf.estimator.Estimator(model_fn=cnnModel,
		model_dir='/model/')

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=50)

	# train the model
	trainInputFunc = tf.estimator.inputs.numpy_input_fn(
		x={'x': images[:50000, :]},
		y=imageLabels[:50000],
		batch_size = 100,
		num_epochs = None,
		shuffle = True)

	glassesClassifier.train(
		input_fn = trainInputFunc,
		steps = 10000,
		hooks = [logging_hook])

	# evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={'x':test_images},
		y=test_labels,
		num_epochs=1,
		shuffle=False)
	eval_results = glassesClassifier.evaluate(input_fn = eval_input_fn)
	print(eval_results)

if __name__ == '__main__':
	tf.app.run()