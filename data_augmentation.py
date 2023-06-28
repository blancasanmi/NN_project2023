import os
import cv2
import numpy as np
import random
import glob
import math
import shutil

TARGET_PATH = 'augmented_data'
TRAINING_PATH = 'training_data\\train'
RANDOM_SEED = 42
SPLIT_PERCENTAGE = 0.8
LABELS = ['A', 'B']


def clear_all_images(image_directory=TARGET_PATH + '\\**\\*.jpg'):
	"""
	Removes all images from the target folders to prevent duplicates or other unexpected behavior.
	"""
	image_paths = glob.glob(image_directory, recursive=True)
	for image_path in image_paths:
		os.remove(image_path)


def split_data(image_directory='training_data\\**\\*.jpg'):
	"""
	Obtains all images, shuffles these, and performs an 80:20 split.
	"""
	training_set = glob.glob(image_directory, recursive=True)
	random.seed(RANDOM_SEED)
	random.shuffle(training_set)
	split = math.ceil(SPLIT_PERCENTAGE * len(training_set))
	reduced_training_set = training_set[:split]
	validation_set = training_set[split:]
	for image in reduced_training_set:
		shutil.copy(image, TARGET_PATH + '\\' + image)
	for image in validation_set:
		shutil.copy(image, TARGET_PATH + '\\' + image.replace('training', 'validation'))


def add_noise(directory, image):
	"""
	Adds noise to the image by removing its blue channel.
	Source: https://pythonexamples.org/python-opencv-remove-blue-channel-from-color-image/
	"""
	src = cv2.imread(directory + image, cv2.IMREAD_UNCHANGED)
	src[:, :, 0] = np.zeros([src.shape[0], src.shape[1]])
	cv2.imwrite('augmented_data\\' + directory + 'distorted_' + image, src)


def augment_data(number_of_augmented_images=500):
	"""
	Augments randomly chosen images from both classes.
	:param number_of_augmented_images: the number of images to augment
	"""
	for label in LABELS:
		images = os.listdir(TRAINING_PATH + label)
		random.seed(RANDOM_SEED)
		random_images = random.sample(images, number_of_augmented_images)
		for image in random_images:
			add_noise(TRAINING_PATH + label + '\\', image)


def main():
	clear_all_images()
	split_data()
	augment_data()


if __name__ == '__main__':
	main()
