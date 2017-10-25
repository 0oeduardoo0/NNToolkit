from simplenn import perceptron

with open('students_data.txt') as f:
	data = f.readlines()

data = [x.strip().split(' ') for x in data]

training_data = []

for d in data[:-5]:
	training_data.append((d[:-1], d[-1:][0]))

print training_data

NN = sperceptron.SimplePerceptron(5)

NN.SetTrainingIterations(10000)
NN.SetTrainingData(training_data)
NN.SetTestData(data[-5:])

#NN.ShowDecisionBoundary(-10, 10)
NN.Learn()
NN.Test()
#NN.ShowDecisionBoundary(-10, 10)