import math
import random

RANDOM_SEED = 87


class LogisticModel:
	""" Represents a logistic model, defined by its model parameters (weights and bias).
	The following references were used for obtaining some more mathematical background (partial derivatives)
	on logistic regression:
	- Jurafsky, D. & Martin, J. (2023). Speech and Language Processing: Logistic Regression.
	https://www.web.stanford.edu/~jurafsky/slp3/5.pdf
	- Stanford University (2019). Section 3_soln. https://cs230.stanford.edu/fall2018/section_files/section3_soln.pdf
	"""
	def __init__(self, weights, bias):
		"""
		Creates a logistic model.
		:param weights: The initial weights vector.
		:param bias: The initial bias.
		"""
		self.__weights = weights
		self.__bias = bias

	def get_parameters(self):
		""" Returns the parameters of the logistic model. """
		return self.__weights.copy(), self.__bias

	def __predict(self, image_vector):
		"""
		Computes the predicted probability given an image vector.
		"""
		logits = inner_product(self.__weights, image_vector) + self.__bias
		return logistic_sigmoid(logits)

	def __compute_gradient(self, data_point, predicted_probability):
		"""
		Computes the gradient by comparing the predicted probability with the teacher output.
		We use Stochastic Gradient Descent.
		"""
		weight_gradient = []
		"""
		Normally I do not put comments in text, but this is pretty remarkable:
		the term 'p - y' (the difference between predicted probability and the true label) is derived from the partial 
		derivative of the cost function with respect to the input to the output neuron. This is then useful for the 
		chain rule.
		"""
		for i in range(len(self.__weights)):
			weight_gradient.append((predicted_probability - data_point.get_label()) * data_point.get_vector()[i])
		bias_gradient = predicted_probability - data_point.get_label()
		return weight_gradient, bias_gradient

	def __update_parameters(self, gradient, learning_rate):
		"""
		Updates the model parameters based on the gradient and the learning rate.
		:param gradient: the gradient vector
		:param learning_rate: the learning rate
		"""
		for i in range(len(self.__weights)):
			self.__weights[i] -= learning_rate * gradient[0][i]
		self.__bias -= learning_rate * gradient[1]

	def evaluate(self, evaluation_set, loss_function):
		"""
		Evaluates the model based on a loss function.
		:param evaluation_set: the set to evaluate the model on.
		:param loss_function: the loss function to evaluate the model with.
		:return: the mean evaluation loss.
		"""
		total_loss = 0
		true_cases = 0
		for data_point in evaluation_set:
			predicted_probability = self.__predict(data_point.get_vector())
			total_loss += loss_function(predicted_probability, data_point.get_label())
			rounded_probability = round(predicted_probability)  # binary decision with threshold 0.5
			true_cases += rounded_probability == data_point.get_label()
		return total_loss / len(evaluation_set), true_cases / len(evaluation_set)

	def train(self, training_set, validation_set, loss_function, learning_rate=0.001, num_epochs=10):
		"""
		Trains the logistic model by iteratively updating the model parameters.
		:param training_set: the set to train the model on.
		:param validation_set: the hold-out validation set to validate the model on.
		:param loss_function: the function to evaluate the model with.
		:param learning_rate: the speed with which the gradient descent is performed.
		:param num_epochs: the number of epochs to train the model.
		"""
		# fix the random seed for reproducibility
		random.seed(RANDOM_SEED)
		for epoch in range(num_epochs):
			random.shuffle(training_set)
			train_loss = 0
			train_true_cases = 0
			for data_point in training_set:
				# Step 1: FP
				predicted_probability = self.__predict(data_point.get_vector())
				# Step 2: BP with SGD
				gradient = self.__compute_gradient(data_point, predicted_probability)
				# Step 3: Parameter updates
				self.__update_parameters(gradient, learning_rate)
				# Step 4:... profit? First compute some training statistics
				train_loss += loss_function(predicted_probability, data_point.get_label())
				rounded_probability = round(predicted_probability)
				train_true_cases += rounded_probability == data_point.get_label()
			val_loss, val_accuracy = self.evaluate(validation_set, loss_function)
			print(f'Epoch {epoch + 1}:')
			print(f'\ttrain loss = {train_loss / len(training_set)}, train accuracy = {train_true_cases / len(training_set)}')
			print(f'\tval loss = {val_loss}, val accuracy = {val_accuracy}')
			print()


def logistic_sigmoid(x):
	"""
	Computes the logistic sigmoid. The logistic sigmoid function is described by s(x) = 1 / (1 + e ** -x).
	"""
	# Bug fix: Prevent overflow issues, so round very low p to 0.
	if x < -200:
		return 0
	return 1 / (1 + math.exp(-x))


def inner_product(vector1, vector2):
	"""
	Using a little bit of Linear Algebra to compute the inner/dot/scalar product of 2 vectors.
	"""
	if len(vector1) != len(vector2):
		raise ValueError(f'The vectors have different dimensions: {len(vector1)} and {len(vector2)}')
	scalar = 0
	for i in range(len(vector1)):
		scalar += vector1[i] * vector2[i]
	return scalar


def cross_entropy_loss(p, y, epsilon=10**-8):
	"""
	Computes the log loss.
	:param p: the predicted probability of the label.
	:param y: the true label.
	:param epsilon: an epsilon used to deal with FPA.
	:return: the log loss.
	"""
	# Bug fix: For FPE with the domain of log()
	if p == 0 or p == 1:
		p = abs(p - epsilon)
	if 0 < p < 1:
		return -math.log(p) if y else -math.log(1 - p)
	raise ValueError(f'Probability {p} out of range (0,1)! How can you be so certain?')
