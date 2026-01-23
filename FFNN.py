import numpy
import scipy.special
import matplotlib.pyplot
import random


class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, weightih (input to hidden) and weightho (hidden to output)
        self.weightih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.weightho = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate

        # sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        # makes the input list a 2D array
        inputs = numpy.array(inputs_list, ndmin=2).T  # this needs to equal (inputnodes, 1)
        targets = numpy.array(targets_list, ndmin=2).T

        # calculates the inputs going into the hidden layer
        hidden_inputs = numpy.dot(self.weightih, inputs)
        # calculates the outputs coming from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculates the inputs going into the output layer
        final_inputs = numpy.dot(self.weightho, hidden_outputs)
        # calculates the outputs (output) coming from the output layer
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.weightho.T, output_errors)

        # update weights (subtract gradient for gradient descent)
        self.weightho += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
        self.weightih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)

        pass

    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.weightih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weightho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


def runMINST():
    input_nodes = 784  # MNIST images are 28x28 = 784 pixels
    hidden_nodes = 200
    output_nodes = 10

    learning_rate = 0.5

    # create instance of neural network
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # load the mnist training data CSV file into a list
    training_data_file = open("mnist_train_100.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train the neural network
    epochs = 700
    for i in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            # make sure that this matches the
            inputs = (numpy.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            pass
        pass

    testing_data_file = open("mnist_test_10.csv", 'r')
    testing_data_list = testing_data_file.readlines()
    testing_data_file.close()

    scorecard = []

    for record in testing_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print(correct_label, "correct label")
        inputs = numpy.asarray(all_values[1:], dtype=numpy.float64) / 255.0 * 0.99 + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        print(label, "networks answer")
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass

    print(scorecard)
    scorecard_array = numpy.asarray(scorecard, dtype=numpy.float64)
    print("MINST performance = ", scorecard_array.sum() / scorecard_array.size)


def runHandwrittenLetters():
    input_nodes = 64  # Handwritten letters are 8x8 = 64 pixels
    hidden_nodes = 200
    output_nodes = 10

    initial_learning_rate = 0.3  # Slightly lower for more stable learning
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, initial_learning_rate)

    # load the mnist training data CSV file into a list
    training_data_file = open("digits-train.txt", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train the neural network
    epochs = 1000  # More epochs with learning rate decay
    print("Training neural network...")
    for i in range(epochs):
        # Shuffle training data each epoch for better learning
        shuffled_data = training_data_list.copy()
        random.shuffle(shuffled_data)

        # Learning rate decay: reduce learning rate over time
        decay_factor = 1.0 - (i / epochs) * 0.5  # Reduce to 50% by end
        n.lr = initial_learning_rate * decay_factor

        for record in shuffled_data:
            all_values = record.strip().split(',')
            # make sure that this matches the - label is at the END, not the beginning
            inputs = (numpy.asarray(all_values[:-1], dtype=float) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[-1])] = 0.99
            n.train(inputs, targets)
            pass

        # Print progress every 100 epochs
        if (i + 1) % 100 == 0:
            print(f"Epoch {i + 1}/{epochs} completed (learning rate: {n.lr:.4f})")
        pass

    testing_data_file = open("digits-test.txt", 'r')
    testing_data_list = testing_data_file.readlines()
    testing_data_file.close()

    scorecard = []

    print("\nTesting neural network...")
    for record in testing_data_list:
        all_values = record.strip().split(',')
        correct_label = int(all_values[-1])
        inputs = numpy.asarray(all_values[:-1], dtype=numpy.float64) / 255.0 * 0.99 + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass

    scorecard_array = numpy.asarray(scorecard, dtype=numpy.float64)
    performance = scorecard_array.sum() / scorecard_array.size
    print(f"\nHandwritten Letters performance = {performance:.4f} ({performance * 100:.2f}%)")


training_data = [
    ([0.01, 0.01], [0.01]),
    ([0.01, 0.99], [0.01]),
    ([0.99, 0.01], [0.01]),
    ([0.99, 0.99], [0.99])
]


def runAND():
    input_nodes = 2
    hidden_nodes = 3
    output_nodes = 1
    learning_rate = 0.5
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    maxEpochs = 2000
    # change to max number of epochs
    for i in range(maxEpochs):
        for inputs, targets in training_data:
            n.train(inputs, targets)
            # test on validation data

            # use a function that computes the accuracy
        pass

    print("\nTesting AND gate:")
    print("Input1 | Input2 | Expected | Output  | Correct")
    print(f'-' * 47)

    scorecard = []
    test_cases = [
        ([0.01, 0.01], 0, "0 AND 0"),
        ([0.01, 0.99], 0, "0 AND 1"),
        ([0.99, 0.01], 0, "1 AND 0"),
        ([0.99, 0.99], 1, "1 AND 1")
    ]

    for inputs, expected, description in test_cases:
        outputs = n.query(inputs)
        output_value = outputs[0][0]
        predicted = 1 if output_value > 0.5 else 0
        input1 = 1 if inputs[0] > 0.5 else 0
        input2 = 1 if inputs[1] > 0.5 else 0
        is_correct = (predicted == expected)
        scorecard.append(1 if is_correct else 0)

        print(f'{input1:5}  |{input2:5}   |{expected:5}     |{predicted:5}    |{is_correct:5}')
        print(f'-' * 47)

        pass

    scorecard_array = numpy.asarray(scorecard, dtype=numpy.float64)
    print(f"AND Performance = {scorecard_array.sum() / scorecard_array.size * 100:.1f}%")


def runXOR():
    input_nodes = 2
    hidden_nodes = 4
    output_nodes = 1
    learning_rate = 0.5

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    epochs = 2000
    for i in range(epochs):
        for inputs, targets in training_data:
            n.train(inputs, targets)
        pass

    print("\nTesting XOR gate:")
    print("Input1 | Input2 | Expected | Output  | Correct")

    scorecard = []
    test_cases = [
        ([0.01, 0.01], 0, "0 XOR 0"),
        ([0.01, 0.99], 1, "0 XOR 1"),
        ([0.99, 0.01], 1, "1 XOR 0"),
        ([0.99, 0.99], 0, "1 XOR 1")
    ]

    for inputs, expected, description in test_cases:
        outputs = n.query(inputs)
        output_value = outputs[0][0]
        predicted = 1 if output_value > 0.5 else 0
        input1 = 1 if inputs[0] > 0.5 else 0
        input2 = 1 if inputs[1] > 0.5 else 0
        is_correct = (predicted == expected)
        scorecard.append(1 if is_correct else 0)

        print(f'{input1:5}  |{input2:5}   |{expected:5}     |{predicted:5}    |{is_correct:5}')
        print(f'-' * 47)

        pass

    scorecard_array = numpy.asarray(scorecard, dtype=numpy.float64)
    print(f"XOR Performance = {scorecard_array.sum() / scorecard_array.size * 100:.1f}%")


# runAND()
# runXOR()
runHandwrittenLetters()
# runMINST()

'''
image_array = numpy.asarray(all_values[1:], dtype = numpy.float64).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='gray', interpolation='none')
matplotlib.pyplot.show()

n.query((numpy.asarray(all_values[1:], dtype = numpy.float64)/255.0*0.99)+0.01)
'''
