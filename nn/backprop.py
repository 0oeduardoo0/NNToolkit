

class Backpropagation:

	def __init__(self, nn):
		self.Network = nn

	def UpdateNodeWeights(self, Node):
		for i in xrange(len(weights)):
			Node.weights[i] -= self.Network.lcoef * Node.inputVector[i] * Node.error

	def UpdateOutputLayerErrors(self):
		i = 0
		for Node in Network.OutputLayer:
			Node.error