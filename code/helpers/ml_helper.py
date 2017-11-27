from tensorflow.examples.tutorials.mnist import input_data
from scipy.cluster.vq import kmeans2
import numpy as np

def readMNISTData():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	return [mnist.train.images, mnist.train.labels,
			mnist.validation.images, mnist.validation.labels,
			mnist.test.images, mnist.test.labels]

# compute cluster centers and labels using kmeans
def applyKmeans2(data, numClusters):
	centroids, labels = kmeans2(data, numClusters, 
		iter=20, minit='points', missing='warn')
	return [centroids, labels]

# compute inverse of spreads for each clusters in the data
def computeClusterSpreadsInvs(data, lbls):
	lbls = np.array(lbls)
	data = np.array(data)

	unique_lbls = np.unique(lbls)
	
	numlbls = unique_lbls.shape[0]
	spreadInvCols = data.shape[1]
	spreadInvs = np.empty([numlbls, spreadInvCols, spreadInvCols])

	for lbl in unique_lbls:
		lbl_indices = np.nonzero(lbls == lbl)
		lbl_cluster = data[lbl_indices]

		var = np.var(lbl_cluster, axis=0)
		spread = var * np.identity(lbl_cluster.shape[1])
		spreadInv = np.linalg.pinv(spread)

		spreadInvs[lbl, :, :] = spreadInv

	return spreadInvs

# compute design matrix of data
def computeDesignMatrixUsingGaussianBasisFunction(data, means, 
	spreadInvs):
	numDataRows = data.shape[0]
	numBasis = means.shape[0]

	designMatrix = np.empty([numDataRows, numBasis])

	for i in range(0, numBasis - 1):
		mean = means[i, :]
		spreadInv = spreadInvs[i, :, :]

		distFromMean = data - mean
		firstBasis = np.sum(np.multiply(
			np.matmul(distFromMean, spreadInv), distFromMean), \
			axis=1)
		firstBasis = np.exp(-0.5 * firstBasis)
		designMatrix[:, i] = firstBasis

	return np.insert(designMatrix, 0, 1, axis=1)

def computeWeightsSetUsingSGD(designMatrix, ouputData, learningRate,
	epochs, batchSize, l2Lambda):
	N,M = designMatrix.shape
	K = ouputData.shape[1]

	weights = np.zeros([K, M])

	for epoch in range(epochs):
		#print('epoch', epoch)
		for i in range(int(N/batchSize)):
			# determine the inputs/outputs in batch
			lowerBound = i * batchSize
			upperBound = min((i + 1) * batchSize, N)

			phi = designMatrix[lowerBound:upperBound, :]
			target = ouputData[lowerBound:upperBound, :]

			# predict class for batch
			predictedClasses = predictClass(phi, weights)

			# compute error gradient for each class
			errorGradients = computeErrorGradient(phi, 
				predictedClasses, target)

			# add regularizer
			error = (errorGradients + (l2Lambda * weights))/\
				len(range(lowerBound, upperBound))

			# update weights
			weights = weights - learningRate * error

	return weights

def computeWeightsUsingStochasticGradientDescentTake2(designMatrix,
	outputData, learningRate, epochs, batchSize, l2Lambda):
	N, M = designMatrix.shape
	K = outputData.shape[1]
	weights = np.random.rand(M, K)
	print('shape of weights matrix:', weights.shape)

	for epoch in range(epochs):
		print('epoch:', str(epoch))
		for i in range(int(N/batchSize)):
			lowerBound = i * batchSize
			upperBound = min((i + 1) * batchSize, N)

			y = np.matmul(designMatrix[lowerBound:upperBound, :],
				weights)

			y = np.exp(y)

			y[np.isnan(y)] = 0
			y[np.isneginf(y)] = -1
			y[np.isposinf(y)] = 1

			y_sum = np.sum(y, axis=1)
			y_sum = np.reshape(y_sum, [y_sum.shape[0], 1])

			y = y/y_sum
			#y = representPredictionProbsAsOneHotVector(y)

			dy = y - outputData[lowerBound:upperBound, :]

			delta_e = np.zeros(weights.shape)

			for dyrow, inputrow in zip(dy,
				designMatrix[lowerBound:upperBound, :]):
				inputrow = np.reshape(inputrow, [inputrow.shape[0], 1])

				delta_e += (dyrow * inputrow)

			delta_e = delta_e / (len(range(lowerBound,upperBound)))

			weights -= learningRate * delta_e

	return weights.T

def predictClass(data, weights):
	predictedClasses = np.zeros([data.shape[0], weights.shape[0]])

	rowIndex = 0
	for singleData in data:
		classProbNum = np.sum(np.multiply(singleData, weights),
			axis=1)
		if np.sum(weights) > 0:
			classProbNum = classProbNum/np.sum(classProbNum)
		predictedClasses[rowIndex, :] = classProbNum
		rowIndex = rowIndex + 1

	return predictedClasses

def computeErrorGradient(data, predictedClasses, target):
	print('inside computeErrorGradient function')
	errorGradients = np.zeros([predictedClasses.shape[1],
		data.shape[1]])

	diff = predictedClasses - target
	print('shape of diff:', diff.shape)

	for i in range(predictedClasses.shape[1]):
		diffCol = np.reshape(diff[:, i],
			[predictedClasses.shape[0], 1])
		errorGradients[i, :] = \
			np.sum(np.multiply(diffCol, data), axis=0)

	return errorGradients

def representPredictionProbsAsOneHotVector(predictionProbs):
	for row in predictionProbs:
		c = np.argmax(row)
		row[:] = 0
		row[c] = 1
	return predictionProbs

def classificationError(predictedClass, actualClass):
	Nwrong = 0
	Ndata = predictedClass.shape[0]

	pCIndex = np.argmax(predictedClass, axis=1)
	aCIndex = np.argmax(actualClass, axis=1)

	diff = pCIndex - aCIndex
	correctMatches = (np.nonzero(diff == 0))[0].shape[0]
	Nwrong = Ndata - correctMatches

	return Nwrong/Ndata