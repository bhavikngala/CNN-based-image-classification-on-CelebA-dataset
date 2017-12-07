from helpers import celebHelper as zeenat
from helpers import image_helper as davinci
from helpers import fileHelper as owl
from helpers import imageAugmentor as leonardo

import numpy as np
import tensorflow as tf

def main():
	imageNameAndLabelsFile = './../data/list_attr_celeba.txt'
	[imageNames, imageLabels] = zeenat.readNamesImageLabels(
		imageNameAndLabelsFile, [15], 2)
	imageLabels = np.asarray(imageLabels, dtype=np.int32)

	eyeglassesIndices = (imageLabels == 1)
	print(eyeglassesIndices.shape)
	
	eyeglassesImageNames = \
		[imageNames[i] for i in range(len(imageNames)) if eyeglassesIndices[i]]

	print(eyeglassesImageNames[0])
	directory = 'D:/datasets/celebA/img_align_celeba/'

	imgs = davinci.batchReadImages(directory, eyeglassesImageNames)
	print(imgs.shape)
	# davinci.showImage(imgs[0])

	rotatedImages = leonardo.rotateImages2(imgs, [0, 45, 30, 15, -15, -30, -45], 'bilinear')
	print(rotatedImages.shape)

	images = []
	for i in range(rotatedImages.shape[0]):
		images.append((davinci.resizeImage(rotatedImages[i, :, :], [100, 100], 'bilinear')).flatten())

	owl.writeNumpyArrayToFile('./../data/numpyarray/', 'eyeglassesAugmented100.npy', np.array(images))

	images = []
	for i in range(rotatedImages.shape[0]):
		images.append((davinci.resizeImage(rotatedImages[i, :, :], [150, 150], 'bilinear')).flatten())

	owl.writeNumpyArrayToFile('./../data/numpyarray/', 'eyeglassesAugmented150.npy', np.array(images))

if __name__ == '__main__':
	main()