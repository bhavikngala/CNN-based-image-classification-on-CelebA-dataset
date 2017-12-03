import numpy as np

def readNamesImageLabels(filename, columnNumbers=None, skipRows=None):
	imageNames = []
	imageLabels = []
	with open(filename) as file:
		lines = file.readlines()
		startRow = 0

		# skip header rows
		if skipRows is not None:
			startRow = skipRows
		for line in lines[startRow:]:
			cols = line.split()

			# append name of image to list
			imageNames.append(cols[0])

			# append labels for that image to the list
			if columnNumbers is not None:
				imageLabels.append([int(cols[index]) for index in columnNumbers])
			else:
				imageLabels.append([int(cols[index]) for index in range(1, len(cols))])

	return [imageNames, imageLabels]

def separateDataSets(images, labels):
	N = images.shape[0]
	p = np.random.permutation(N)

	return [images[p[:int(N*0.8)]], labels[p[:int(N*0.8)]], \
			images[p[int(N*0.8):int(N*0.9)]], labels[p[int(N*0.8):int(N*0.9)]], \
			images[p[int(N*0.9):]], labels[p[int(N*0.9):]]]