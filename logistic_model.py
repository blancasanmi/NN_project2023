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
		return logistic_sigmoid(inner_product(self.__weights, image_vector) + self.__bias)

	def __compute_gradients(self, data_point, predicted_probability):
		"""
		Computes the gradients by comparing the predicted probability with the teacher output.
		"""
		weight_gradients = []
		for i in range(len(self.__weights)):
			weight_gradients.append((predicted_probability - data_point.get_label()) * data_point.get_vector()[i])
		bias_gradient = predicted_probability - data_point.get_label()
		return weight_gradients, bias_gradient

	def __update_model_parameters(self, gradients, learning_rate):
		for i in range(len(self.__weights)):
			self.__weights[i] -= learning_rate * gradients[0][i]
		self.__bias -= learning_rate * gradients[1]

	def train(self, training_set, learning_rate=0.001, num_epochs=10):
		"""
		Trains the logistic model by updating the model parameters.
		"""
		for epoch in range(num_epochs):
			total_loss = 0
			for data_point in training_set:
				predicted_probability = self.__predict(data_point.get_vector())
				gradients = self.__compute_gradients(data_point, predicted_probability)
				self.__update_model_parameters(gradients, learning_rate)
				total_loss += cross_entropy_loss(predicted_probability, data_point.get_label())
			print(f'Mean training loss at epoch {epoch + 1}: {total_loss / len(training_set)}')


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


def cross_entropy_loss(p, y):
	"""
	Computes the log loss.
	:param p: the predicted probability of the label.
	:param y: the true label.
	:return: the log loss.
	"""
	if 0 < p < 1:
		return -math.log(p) if y else -math.log(1 - p)
	return 0 if p == y else 100000  # very large loss
	# maybe introduce an epsilon for working with FPA
	# raise ValueError(f'Probability {p} out of range (0,1)!')
