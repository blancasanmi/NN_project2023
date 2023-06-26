from PIL import Image
import os


class DataPoint:
	def __init__(self, vector, label):
		self.vector = vector
		self.label = label


TRAINING_PATH = 'training_data\\train'
TEST_PATH = 'test_data\\test'
RESCALING_CONSTANT = 255
DEBUG_MAX_IMAGES = 10  # we use this to speed up the data reading, remove later


def image_to_vector(image):
	""" Converts an image into a 1D vector.
	:param image: the image to be converted
	:return: the 1D vector corresponding to the image, where the channels are flattened.
	"""
	vector = []
	for pixel in image.getdata():
		vector.extend([channel / RESCALING_CONSTANT for channel in pixel])
	return vector


def get_labelled_set(folder_path, label):
	""" This method is rather slow. Could be improved with multithreading or batch processing. Or, directly using it to
	train to save memory.
	:param folder_path: the folder where the data set is located.
	:param label: the label of the data set.
	:return: the labelled set.
	"""
	data_set = []
	debug_image_index = 0
	for filename in os.listdir(folder_path):
		if debug_image_index == DEBUG_MAX_IMAGES:
			break
		debug_image_index += 1
		fp = os.path.join(folder_path, filename)
		with Image.open(fp) as image:
			data_set.append(DataPoint(image_to_vector(image), label))
	return data_set


def get_entire_set(path):
	""" Returns the training set or test set.
	:param path: the path to the folder containing the set.
	:return: the training or test set.
	"""
	data_set = []
	unlabelled_folder_path = os.getcwd() + '\\' + path
	for label in ['A', 'B']:
		data_set.extend(get_labelled_set(unlabelled_folder_path + label + '\\', label))
	return data_set


def main():
	training_set = get_entire_set(TRAINING_PATH)
	# train logit model on training set
	# load test set
	# test logit model on test set


if __name__ == "__main__":
	main()