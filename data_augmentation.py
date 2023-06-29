"""
Note: this module makes assumptions about the folder structure. This is more or less unavoidable with data manipulation.
Put this module in a sensible place. The structure should look as follows:
augmented_data
	training_data
		trainA // empty
		trainB // empty
	validation_data
		trainA // empty
		trainB // empty
training_data
	trainA // training images
	trainB // training images

Good luck!
Dries
"""
import os
import cv2
import numpy as np
import random
import glob
import math
import shutil

TARGET_PATH = 'augmented_data'
TRAINING_PATH = 'training_data\\train'
RANDOM_SEED_1 = 42
RANDOM_SEED_2 = 21
SPLIT_PERCENTAGE = 0.8
LABELS = ['A', 'B']


def clear_all_images(image_directory=TARGET_PATH + '\\**\\*.jpg'):
	"""
	Removes all images from the target folders to prevent duplicates or other unexpected behavior.
	:param image_directory: the path to the folder containing the reduced training set and validation set.
	"""
	image_paths = glob.glob(image_directory, recursive=True)
	for image_path in image_paths:
		os.remove(image_path)


def split_data(image_directory='training_data\\**\\*.jpg'):
	"""
	Obtains all images, shuffles these, performs an 80:20 split, and copies them to another directory.
	:param image_directory: the directory which contains the entire training set.
	"""
	training_set = glob.glob(image_directory, recursive=True)
	random.seed(RANDOM_SEED_1)
	random.shuffle(training_set)
	split = math.ceil(SPLIT_PERCENTAGE * len(training_set))  # Keras also uses ceiling
	reduced_training_set = training_set[:split]
	validation_set = training_set[split:]
	for image in reduced_training_set:
		shutil.copy(image, TARGET_PATH + '\\' + image)
	for image in validation_set:
		shutil.copy(image, TARGET_PATH + '\\' + image.replace('training', 'validation'))


def transform_image(directory, image):
	"""
	Transforms the image by removing its blue channel.
	Source: https://pythonexamples.org/python-opencv-remove-blue-channel-from-color-image/
	:param directory: the directory to read the image from.
	:param image: the image to transform.
	"""
	src = cv2.imread(directory + image, cv2.IMREAD_UNCHANGED)
	src[:, :, 0] = np.zeros([src.shape[0], src.shape[1]])
	cv2.imwrite(directory + 'distorted_' + image, src)


def augment_data(number_of_augmented_images=500):
	"""
	Augments randomly chosen images from both classes.
	:param number_of_augmented_images: the number of images to augment
	"""
	path = TARGET_PATH + '\\' + TRAINING_PATH
	for label in LABELS:
		images = os.listdir(path + label)
		random.seed(RANDOM_SEED_2)
		random_images = random.sample(images, number_of_augmented_images)
		for image in random_images:
			transform_image(path + label + '\\', image)


def main():
	clear_all_images()
	split_data()
	augment_data()


if __name__ == '__main__':
	main()
