from math import pi
import tensorflow as tf
import numpy as np

def rotateImages(images, angles, imageRes, nChannels):
	rotatedImages = []

	# placeholder for images
	_images = tf.placeholder(tf.float32, \
		[[-1, imageRes[0], imageRes[1], nChannels]])
	# placeholder for rotation angles
	_rotateAngels = tf.placeholder(tf.float32, \
		shape=(len(images)))

	# tensorflow rotation function
	rotation_fn = tf.contrib.image.rotate(_images, \
		_rotateAngels)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for angle in angles:
			rotationAngleInRadians = \
				[angle * pi / 180] * len(images)

			rotatedImages = sess.run(rotation_fn, 
				feed_dict={_images:images, \
					_rotateAngels:rotationAngleInRadians})

			rotatedImages.extends(rotatedImages)

	rotatedImages = np.array(rotatedImages, np.float32)
	return rotatedImages