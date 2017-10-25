from simplenn import perceptron

# ([calificacion, faltas], bueno | malo)
training_data = [
	([10, 1], 1),
	([9.3, 2], 1),
	([8.5, 2], 1),
	([10, 3], 1),
	([9.2, 0], 1),
	([8.5, 1], 1),
	([10, 2], 1),
	([8.3, 0], 1),
	([8.8, 3], 1),
	([10, 0], 1),
	([8, 3], 1),
	([8.6, 2], 1),

	([5, 8], 0),
	([6, 3], 0),
	([7.4, 5], 0),
	([7, 2], 0),
	([5, 4], 0),
	([6.2, 4], 0),
	([7.4, 8], 0),
	([5, 1], 0),
	([4.6, 4], 0),
	([7.7, 3], 0),
	([7.1, 6], 0),
	([6.5, 3], 0),
	([6.7, 8], 0)
]

test_data = [
	[10, 4],
	[7.5, 0],
	[9.3, 2],
	[6.5, 9],
	[5, 3]
]

NN = perceptron.SimplePerceptron(2)

NN.SetTrainingIterations(1000)
NN.SetTrainingData(training_data)
NN.SetTestData(test_data)

NN.ShowDecisionBoundary(-10, 10)
NN.Learn()
NN.Test()
NN.ShowDecisionBoundary(-10, 10)