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

def main():
	imageNameAndLabelsFile = './../data/list_attr_celeba.txt'
	[imageNames, imageLabels] = zeenat.readNamesImageLabels(
		imageNameAndLabelsFile, [15], 2)

	images = owl.readNumpyArrayFromFile('./../data/numpyarray/images.npy')
	davinci.plotImage(images[0,:], [100, 100])

	# readImages()

if __name__ == '__main__':
	main()