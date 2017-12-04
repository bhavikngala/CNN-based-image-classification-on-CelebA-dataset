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
	directory = './../data/img_align_celeba/'

	imgs = davinci.batchReadImages(directory, [eyeglassesImageNames[0]])
	print(imgs.shape)
	# davinci.showImage(imgs[0])

	rotatedImages = leonardo.rotateImages2(imgs, [45, 30, 15, -15, -30, -45], 'bilinear')

	for img in rotatedImages:
		davinci.showImage(img)

if __name__ == '__main__':
	main()