from simplenn import perceptron

training_data = [
	([0, 0], 0),
	([0, 1], 1),
	([1, 0], 1),
	([1, 1], 1)
]

test_data = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1]
]

NN = sperceptron.SimplePerceptron(2)

NN.SetTrainingIterations(10)
NN.SetTrainingData(training_data)
NN.SetTestData(test_data)

NN.ShowDecisionBoundary(-2, 2)

NN.Learn()
NN.Test()

NN.ShowDecisionBoundary(-2, 2)

#x1 = int(raw_input('Entrada 1: '))
#x2 = int(raw_input('Entrada 2: '))
#print "Salida: %s" % (NN.ComputeOutput([x1, x2]))