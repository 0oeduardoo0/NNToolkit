from random import choice
from numpy import random
import matplotlib.pyplot as plt
from . import tfunctions
import numpy as np

class SimplePerceptron:

	def __init__(self, n_inputs):

		self.weights = []
		self.weights_history = []
		self.training_errors = []
		self.lcoef  = 0.1
		self.iterations = 100
		self.n_inputs = n_inputs + 1 # inputs y el bias
		self.training_data = None
		self.test_data = None
		
		self.weights = random.rand(self.n_inputs)
		for i in xrange(self.n_inputs):
		#	self.weights.append(1)
			self.weights_history.append([1])

	def SetLcoef(self, lcoef):
		self.lcoef = lcoef

	def SetTrainingIterations(self, iterations):
		self.iterations = iterations

	def SetTrainingData(self, data):
		self.training_data = data

	def SetTestData(self, data):
		self.test_data = data

	def AddBiasToInputArray(self, p):
		p_copy = list(p)
		p_copy.append(1) # bias input

		return p_copy

	def UpdateWeights(self, data, error):
		p = self.AddBiasToInputArray(data)

		for i in xrange(self.n_inputs):
			self.weights[i] += self.lcoef * error * float(p[i])
			self.weights_history[i].append(self.weights[i])

	def ComputeOutput(self, data):
		p = self.AddBiasToInputArray(data)
		I = 0

		for j in xrange(self.n_inputs):
			I += self.weights[j] * float(p[j])

		return tfunctions.hard_step(I)

	def Learn(self):
		for i in xrange(self.iterations):
			for data, expected in self.training_data:				
				x = self.ComputeOutput(data)
				error = int(expected) - x

				self.training_errors.append(error)

				self.UpdateWeights(data, error)

	def Test(self):
		for data in self.test_data:
			x  = self.ComputeOutput(data)

			print "p=%s o=%s" % (data, x)

		print "w=%s" % (self.weights[:-1])
		print "b=%s" % (self.weights[-1:])

		plt.plot(self.training_errors)

		for w in self.weights_history:
			plt.plot(w)

		plt.ylabel('Error')
		plt.xlabel('Iteration')
		plt.show()

	def ShowDecisionBoundary(self, x, y):
		h = 0.2

		# Crear una malla
		x_min, x_max = x, y
		y_min, y_max = x, y
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

		fig, ax = plt.subplots()
		values = np.c_[xx.ravel(), yy.ravel()]
		Z = []
		# calcular el resultado de la red para la malla
		for x in values:
			Z.append(self.ComputeOutput(x))

		# Put the result into a color plot
		Z = np.ma.array(Z)
		Z = Z.reshape(xx.shape)
		ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
		ax.axis('off')

		# Plot also the training points
		X1, X2, Y = [], [], []

		for x, y in self.training_data:
			X1.append(x[0])
			X2.append(x[1])
			Y.append(y)

		ax.scatter(X1, X2, c=Y, cmap=plt.cm.Paired)

		ax.set_title('Decision Boundary')
		plt.show()