import os
from PIL import Image
import random
from logistic_model import LogisticModel, cross_entropy_loss

NUMBER_OF_FEATURES = 256 * 256 * 3  # The image dimensions are 256x256 and there are 3 color channels.
ROOT = 'augmented_data\\'
TRAINING_PATH = 'training_data\\train'
VALIDATION_PATH = 'validation_data\\train'
TEST_PATH = 'test_data (do not open!!!1!!!)\\test'
LABELS = ['A', 'B']
TRUE_LABEL = LABELS[0]  # arbitrary choice
RESCALING_CONSTANT = 255
DEBUG_MAX_IMAGES = 1000000000  # we use this to speed up the data reading, remove later
RANDOM_SEED = 87


class DataPoint:
	""" Represents a data point which consists of a vector representation of the image and its label. """
	def __init__(self, vector, label, filename):
		"""
		Creates a data point.
		:param vector: the image vector.
		:param label: the image label.
		"""
		if len(vector) != NUMBER_OF_FEATURES:
			raise ValueError(f'Unexpected dimension: {len(vector)}')
		self.__vector = vector
		self.__label = label
		self.__name = filename

	def get_vector(self):
		"""
		Returns the 1D vector associated with the image.
		:return: the image vector.
		"""
		return self.__vector

	def get_label(self):
		"""
		Returns the image label.
		:return: the image label.
		"""
		return self.__label

	def get_name(self):
		"""
		Returns the name of the image. Useful for debugging.
		"""
		return self.__name


def image_to_vector(image):
	""" Flattens an image into a 1D vector.
	:param image: the image to be flattened.
	:return: the 1D vector corresponding to the image, where the channels are flattened.
	"""
	vector = []
	for pixel in image.getdata():
		vector.extend([channel / RESCALING_CONSTANT for channel in pixel])
	return vector


def get_labelled_set(folder_path, label):
	""" This method is rather slow. Could be improved with multithreading or batch processing.
	:param folder_path: the folder where the data set is located.
	:param label: the label of the data set.
	:return: the labelled set.
	"""
	labelled_data_set = []
	debug_image_index = 0
	for filename in os.listdir(folder_path):
		if debug_image_index == DEBUG_MAX_IMAGES:
			break
		debug_image_index += 1
		fp = os.path.join(folder_path, filename)
		with Image.open(fp) as image:
			labelled_data_set.append(DataPoint(image_to_vector(image), label == TRUE_LABEL, filename))
	return labelled_data_set


def get_entire_data_set(path):
	""" Returns the training set or test set.
	:param path: the path to the folder containing the set.
	:return: the training or test set.
	"""
	data_set = []
	unlabelled_folder_path = os.getcwd() + '\\' + path
	for label in LABELS:
		data_set.extend(get_labelled_set(unlabelled_folder_path + label + '\\', label))
	return data_set


def train_model(model):
	training_set = get_entire_data_set(ROOT + TRAINING_PATH)
	validation_set = get_entire_data_set(ROOT + VALIDATION_PATH)
	print(f'loaded training ({len(training_set)} images) and validation set ({len(validation_set)} images)')
	# seems useful to shuffle training set, fix the random seed
	random.seed(RANDOM_SEED)
	random.shuffle(training_set)
	model.train(training_set, validation_set, cross_entropy_loss)


def test_model(model):
	test_set = get_entire_data_set(TEST_PATH)
	loss, accuracy = model.evaluate(test_set, cross_entropy_loss)
	print(f'test loss = {loss}, test accuracy = {accuracy}')


def main():
	logistic_model = LogisticModel([0] * NUMBER_OF_FEATURES, 0)
	train_model(logistic_model)
	# test_model(logistic_model)


if __name__ == "__main__":
	main()
