import tensorflow as tf

def readNamesImageLabels(filename, columnNumbers=None, skipRows=None):
	imageNames = []
	imageLabels = []
	with open(filename) as file:
		lines = file.readlines()
		startRow = 0

		# skip header rows
		if skipRows is not None:
			startRow = skipRows - 1
		for line in lines[startRow:]:
			cols = line.split()

			# append name of image to list
			imageNames.append(cols[0])

			# append labels for that image to the list
			if columnNumbers is not None:
				imageLabels.append([int(cols[index]) for index in range(columnNumbers)])
			else:
				imageLabels.append([int(cols[index]) for index in range(len(cols))])

	return [imageNames, imageLabels]