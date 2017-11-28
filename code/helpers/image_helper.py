from scipy import misc
import matplotlib.pyplot as plt
import os
import numpy as np

def readImage(filename):
	return misc.imread(filaname)

def resizeImage(img, outputSize, interpMethod):
	return misc.imresize(img, outputSize, interp = interpMethod)

def showImage(img):
	plt.imshow(img)
	plt.show()

def batchReadAndResizeImages(directory, outputSize, interpMethod, imageExtension):
	# 2D array of images
	imgs = []
	for file in os.listdir(directory):
		if file.endswith(imageExtension):
			img = misc.imread(directory+'/'+file, flatten=True);
			img = resizeImage(img, outputSize, interpMethod)
			#img = 1 - normalizeImage(img)
			img = 1 - img
			img = img.flatten()
			imgs.append(img)
	return imgs

def normalizeImage(img):
	return img/255

def readUSPSTestImagesAndLbls(directory):
	images = []
	lbls = []
	
	images = batchReadAndResizeImages(directory, [28, 28], 'bilinear')

	for i in range(0, 10):
		lbl = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		lbl[i] = 1
		lbl = [lbl] * 150
		lbls = lbl + lbls

	return [images, lbls]

def readUSPSTrainImagesAndLbls(directory):
	images = []
	lbls = []
	
	images = batchReadAndResizeImages(directory, [28, 28], 'bilinear')

	for i in range(0, 10):
		imgs = batchReadAndResizeImages(directory+'/'+str(i),
			[28, 28], 'bilinear')
		images = images + imgs
		lbl = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		lbl[i] = 1
		lbl = [lbl] * len(imgs)
		lbls = lbls + lbl

	return [images, lbls]

# function plots an image, input is flattened image
def plotImage(flatImg):
	img = np.resize(flatImg, [28, 28])
	showImage(img)