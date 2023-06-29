import math


class LogisticModel:
	""" Represents a logistic model, defined by its model parameters (weights and bias).
	The following reference was used for obtaining some more mathematical background on logistic regression:
	Jurafsky, D. & Martin, J. (2023). Speech and Language Processing: Logistic Regression.
	www.web.stanford.edu/~jurafsky/slp3/5.pdf
	"""
	def __init__(self, weights, bias):
		"""
		Creates a logistic model.
		:param weights: The initial weights vector.
		:param bias: The initial bias.
		"""
		self.__weights = weights
		self.__bias = bias

	def __predict(self, image_vector):
		"""
		Computes the predicted probability given an image vector.
		"""
		logits = inner_product(self.__weights, image_vector) + self.__bias
		return logistic_sigmoid(logits)

	def __compute_gradients(self, data_point, predicted_probability):
		"""
		Computes the gradients by comparing the predicted probability with the teacher output.
		"""
		weight_gradients = []
		for i in range(len(self.__weights)):
			weight_gradients.append((predicted_probability - data_point.get_label()) * data_point.get_vector()[i])
		bias_gradient = predicted_probability - data_point.get_label()
		return weight_gradients, bias_gradient

	def __update_parameters(self, gradients, learning_rate):
		"""
		Updates the model parameters based on the gradients and the learning rate.
		"""
		for i in range(len(self.__weights)):
			self.__weights[i] -= learning_rate * gradients[0][i]
		self.__bias -= learning_rate * gradients[1]

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
			rounded_probability = round(predicted_probability)
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
		# TODO maybe use patience rather than a number of epochs? Then see where it stagnates.
		for epoch in range(num_epochs):
			train_loss = 0
			train_true_cases = 0
			for data_point in training_set:
				predicted_probability = self.__predict(data_point.get_vector())
				gradients = self.__compute_gradients(data_point, predicted_probability)
				self.__update_parameters(gradients, learning_rate)
				train_loss += loss_function(predicted_probability, data_point.get_label())
				rounded_probability = round(predicted_probability)
				train_true_cases += rounded_probability == data_point.get_label()
			val_loss, val_accuracy = self.evaluate(validation_set, loss_function)
			print(f'Epoch {epoch + 1}:')
			print(f'\ttrain loss = {train_loss}, train accuracy = {train_true_cases / len(training_set)}')
			print(f'\tval loss = {val_loss}, val accuracy = {val_accuracy}')
			print()


def logistic_sigmoid(x):
	"""
	Computes the logistic sigmoid. The logistic sigmoid function is described by s(x) = 1 / (1 + e ** -x).
	"""
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
	if p == 0 or p == 1:
		p = abs(p - epsilon)
	if 0 < p < 1:
		return -math.log(p) if y else -math.log(1 - p)
	raise ValueError(f'Probability {p} out of range (0,1)! How can you be so certain?')
