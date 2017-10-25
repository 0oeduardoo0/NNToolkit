

class Neuron:

	def __init__(self):
		self.nodeType = None
		self.weights = []
		self.inputVector = []
		self.bias = 1
		self.activationFunction = None
		self.dActivationFunction = None
		self.ponderateSum = 1
		self.error = 1
		self.outputState = 1

	def ComputeOutput(self, x):
		self.inputVector = list(x)
		self.ponderateSum = (self.weights * self.inputVector).sum()
		self.outputState = self.activationFunction(self.ponderateSum)