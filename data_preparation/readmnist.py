import numpy as np
import sys
from os import path
DATA_DIR = "data"
TRAIN_SET = path.join(DATA_DIR, "train-images-idx3-ubyte")
TRAIN_LABELS = path.join(DATA_DIR, "train-labels-idx1-ubyte")
TEST_SET = path.join(DATA_DIR, "t10k-images-idx3-ubyte")
TEST_LABELS = path.join(DATA_DIR, "t10k-labels-idx1-ubyte")

def read_training(image_limit=sys.maxsize):
    return read_data(TRAIN_SET, TRAIN_LABELS, image_limit=image_limit)

def read_test(image_limit=sys.maxsize):
    return read_data(TEST_SET, TEST_LABELS, image_limit=image_limit)

def read_data(imageFilename, labelFilename, image_limit=sys.maxsize):
	def read_ints(f, numInts):
		result = []
		for i in range(numInts):
			b = f.read(4)
			if b:
				asint = int.from_bytes(b, byteorder="big", signed=False)
				result.append(asint)

		return (result)
	def read_images():
		images = []
		with open(imageFilename, "rb") as f:
			if 2051 != read_ints(f, 1)[0]:
				raise ValueError()

			numImages,numRows,numCols = read_ints(f, 3)

			for image in range(min(numImages, image_limit)):
				b = f.read(numRows*numCols)
				imgdata = list(b)
				images.append(imgdata)

		return images
	def read_labels(image_limit=sys.maxsize):
		labels = []
		with open(labelFilename, "rb") as f:

			if 2049 != read_ints(f, 1)[0]:
				raise Exception()
			numLabels, = read_ints(f, 1)
			numToRead = min(numLabels, image_limit)
			labels = list(f.read(numToRead))

		return labels
	images = read_images()
	labels = read_labels()
	return images, labels

def append_ones_row(arr):
	temp = np.ones((len(arr), len(arr[0])+1))
	temp[:, 1:] = arr
	return temp

def error_rate(labels, prediction):
	correct = 0
	incorrect = 0
	digitsc = [0 for _ in range(10)]
	digitsi = [[0 for _ in range(10)] for _ in range(10)]
	for i in range(len(labels)):
		x = []
		if labels[i] == prediction[i]:
			correct = correct + 1
			digitsc[labels[i]] = digitsc[labels[i]] + 1
		else:
			incorrect = incorrect + 1
			digitsi[labels[i]][prediction[i]] += 1
	# print(np.argmin(digitsi))
	maxLabel = np.argmax(digitsi)/10
	maxLabelV = np.argmax(digitsi)%10

	print("Max failure: ", str(maxLabel), " as ", str(maxLabelV))
	print ("C:", correct, "I:", incorrect, "%:", (correct/(correct+incorrect)))



