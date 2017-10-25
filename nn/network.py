

class NeuralNetwork:

	def __init__(self, lcoef):
		self.lcoef = lcoef
		self.inputVector = None
		self.expectedVector = None
		self.outputVector = None

		self.inputLayer = None
		self.hiddenLayers = list()
		self.outputLayer = None

	def setInputLayer(self, Layer layer):
		self.inputLayer = layer

	def setOutputLayer(self, Layer layer):
		self.outputLayer = layer

	def addHiddenLayer(self, Layer layer):
		self.hiddenLayers.extends(layer)