from helpers import celebHelper as zeenat
from helpers import image_helper as davinci
from helpers import fileHelper as owl
import numpy as np

def readImages():
	imageDirectory = './../data/img_align_celeba/'

	images = davinci.batchReadAndResizeImages(imageDirectory, [150, 150], 'bilinear', '.jpg')
	images = np.array(images)

	davinci.plotImage(images[0,:], [150, 150])

	owl.writeNumpyArrayToFile('./../data/numpyarray/', 'images150.npy', images)

def transformLabels(labels):
	labels[labels>=0.5] = 1
	return (labels[labels<0.5] = 0)

def cnnModel(features, labels, mode):
	'''Model function for CNN.'''
	# input layer
	input_layer = tf.reshape(features['x'], [-1, 150, 150, 1])

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding='same',
		activation=tf.nn.relu)

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(
		inputs=conv1,
		pool_size=[2, 2],
		strides=2)

	# Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5,5],
		padding='same',
		activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(
		inputs=conv2,
		pool_size=[2, 2],
		strides=2)

	# Dense Layers
	pool2_flat = tf.reshape(pool2, [-1, 7*7*64])

	dense = tf.layers.dense(
		inputs=pool2_flat,
		units=1024,
		activation=tf.nn.relu)

	dropout = tf.layers.dropout(
		inputs=dense,
		rate=0.4,
		training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits Layer
	logits = tf.layers.dense(
		inputs=dropout,
		units=1)

	predictions = {
		'classes' = transformLabels(logits.eval()),
		'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
	}



def main():
	imageNameAndLabelsFile = './../data/list_attr_celeba.txt'
	[imageNames, imageLabels] = zeenat.readNamesImageLabels(
		imageNameAndLabelsFile, [15], 2)

	imageLabels = transformLabels(imageLabels)

	images = owl.readNumpyArrayFromFile('./../data/numpyarray/images.npy')
	davinci.plotImage(images[0,:], [100, 100])

	# readImages()

if __name__ == '__main__':
	main()