def readImageLabels(filename, columnNumbers=None, skipRows=None):
	imageLabels = []
	with open(filename) as file:
		lines = file.readlines()
		startRow = 0
		if skipRows is not None:
			startRow = skipRows - 1
		for line in lines[startRow:]:
			cols = line.split()
			if columnNumbers is not None:
				imageLabels.append([cols[index] for index in columnNumbers])
			else:
				imageLabels.append(cols)