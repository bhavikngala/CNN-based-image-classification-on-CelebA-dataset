from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from helpers import celebHelper as zeenat
from helpers import image_helper as davinci
from helpers import fileHelper as owl

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# for image res 50*50
imageRes = [100, 00]
imageResInput = [-1, 100, 100, 1]
pool2outputSize = 25 * 25 * 64


def readImages():
	imageDirectory = './../data/img_align_celeba/'

	images = davinci.batchReadAndResizeImages(imageDirectory, imageRes, 'bilinear', '.jpg')
	images = np.array(images)

	davinci.plotImage(images[0,:], imageRes)

	owl.writeNumpyArrayToFile('./../data/numpyarray/', 'images150.npy', images)

def dataAndSampleItRandomly(imageFilename, labelsFilename, numSamples):
	[_, imageLabels] = zeenat.readNamesImageLabels(
		labelsFilename, [15], 2)

	imageLabels = np.asarray(imageLabels, dtype=np.int32)
	imageLabels[imageLabels==-1] = 0
	imageLabels = imageLabels.flatten()

	images = owl.readNumpyArrayFromFile(imageFilename)
	images = images.astype('float32')

	# randomIndices = np.random.permutation(numSamples)
	N = images.shape[0]

	return [images[0:numSamples], imageLabels[0:numSamples], images[int(N*0.9):], imageLabels[int(N*0.9):]]

def readAugmentedData(filename):
	images = owl.readNumpyArrayFromFile(filename)
	images = images.astype('float32')
	imageLabels = np.zeros([1, images.shape[0]], dtype=np.int32)
	imageLabels = imageLabels.flatten()
	return [images, imageLabels]

def generateTrainingValiTestDatasets(imageFilename, labelsFilename,
		augmentedFilename, numSamples):
	[imagesA, imageLabelsA] = readAugmentedData(augmentedFilename)
	[images, imageLabels, testImages, testLabels] = \
		dataAndSampleItRandomly(imageFilename, labelsFilename, numSamples)
	images = np.concatenate([images, imagesA], axis=0)
	imageLabels = np.concatenate([imageLabels, imageLabelsA])
		
	return [images, imageLabels, testImages, testLabels]

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
	imagesFilename = './../data/numpyarray/images100.npy'
	augmentedFilename = './../data/numpyarray/eyeglassesAugmented100.npy'

	[images, imageLabels, test_images, test_labels] = \
		generateTrainingValiTestDatasets(imagesFilename, imageNameAndLabelsFile, augmentedFilename, 65000)

	malesDir = './modelMales/'
	# eyeglassesDir50 = '/model/'
	eyeglassesDir50 = './model_50/'
	eyeglassesDir100 = './model100/'
	eyeglassesDir150 = './model150/'

	# create the classifier
	glassesClassifier = tf.estimator.Estimator(model_fn=cnnModel,
		model_dir=eyeglassesDir100)

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	# tensors_to_log = {"probabilities": "softmax_tensor"}
	# logging_hook = tf.train.LoggingTensorHook(
		# tensors=tensors_to_log, every_n_iter=50)

	# train the model
	trainInputFunc = tf.estimator.inputs.numpy_input_fn(
		x={'x': images},
		y=imageLabels,
		batch_size = 250,
		num_epochs = None,
		shuffle = True)

	glassesClassifier.train(
		input_fn = trainInputFunc,
		steps = 2500)

	# evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={'x':test_images},
		y=test_labels,
		num_epochs=1,
		shuffle=False)
	eval_results = glassesClassifier.evaluate(input_fn = eval_input_fn)
	print(eval_results)

	verifyImage = test_images[0:10]
	verifyLabels = test_labels[0:10]


	predict_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": verifyImage},
		num_epochs=1,
		shuffle=False)

	predictions = list(glassesClassifier.predict(input_fn=predict_input_fn))
	predicted_classes = [p["classes"] for p in predictions]

	print(
		"New Samples, Class Predictions:    {}\n"
		.format(predicted_classes))

	print('\n\n~~~~~~~~~~~~~~~~~~~~~~ labels\n',verifyLabels)

if __name__ == '__main__':
	tf.app.run()