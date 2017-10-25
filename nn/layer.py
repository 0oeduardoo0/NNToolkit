from .neuron import Neuron

class Layer:

	def __init__(self, nodes):
		self.nodes = list()
		self.type  = None
		self.inputVector = None
		self.outputVector = None

		for i in xrange(nodes):
			self.nodes.append(Neuron())

	def setActivationFunciton(self, f, df):
		for node in self.nodes:
			node.activationFunction = f
			node.dActivationFunction = dfping

	def computeOutput(self):
		for node in self.nodes:
			self.outputVector.append(node.computeOutput())
