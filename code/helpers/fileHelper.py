import numpy as np
import os

def writeNumpyArrayToFile(directory, filename, nparray):
	createDirectory(directory)
	np.save(directory + filename, nparray)
	# print('wrote to file:', directory, filename)

def readNumpyArrayFromFile(filename):
	return np.load(filename)

def directoryExists(directory):
	return os.path.exists(directory)

def createDirectory(directory):
	if not directoryExists(directory):
		os.makedirs(directory)

def saveDataToFile(directory, filename, data):
	createDirectory(directory)
	file = open(directory + filename, 'w')
	for line in data:
		file.write(str(line)+'\n')
	file.close()