#!/usr/bin/env python

import numpy
import scipy.special
import matplotlib.pyplot

class neuralNetwork:
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass
    
    def train(self, inputs_list, targets_list):
        inputs  = numpy.array(inputs_list,  ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        hidden_iputs, hidden_oputs = self.calc_outputs(self.wih, inputs)

        final_iputs, final_oputs = self.calc_outputs(self.who, hidden_iputs)

        output_errors = targets - final_oputs
        
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        self.who += self.lr * numpy.dot((output_errors * final_oputs *  (1.0 - final_oputs)),  numpy.transpose(hidden_oputs))
        
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_oputs * (1.0 - hidden_oputs)), numpy.transpose(inputs))

        pass
    
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        hidden_iputs, hidden_oputs = self.calc_outputs(self.wih, inputs)
        
        final_iputs, final_oputs = self.calc_outputs(self.who, hidden_iputs)
        
        return final_oputs
    
    
    def calc_outputs(self, weights, inputs):
        iputs = numpy.dot(weights, inputs)
        oputs = self.activation_function(iputs)
        
        return iputs, oputs    
    
    

inputnodes   = 784
hiddennodes  = 100
outputnodes  = 10
learningrate = 0.1


n = neuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)

training_data_file = open("MNIST_CSV/mnist_train.csv", "r");
training_data_list = training_data_file.readlines()
training_data_file.close()

epoch = 2

for e in range(epoch):
    
    counter = 1
    
    for record in training_data_list:
        
        print("train epoch " + str(e + 1) + " row " + str(counter))
        
        all_values = record.split(',')
        
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01;
        
        targets = numpy.zeros(outputnodes) + 0.01
        
        targets[int(all_values[0])] = 0.99
                
        n.train(inputs, targets)
        
        counter += 1
        
        pass
    
    pass


test_data_file = open("MNIST_CSV/mnist_test.csv", "r");
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    
    correct_label = int(all_values[0])
    
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01;
    
    outputs = n.query(inputs)
    
    label = numpy.argmax(outputs)
    
    print("calculated label " + str(label) + " vs. " + str(correct_label))
    
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass

    pass

scorecard_array = numpy.asarray(scorecard)

print("performance = ", scorecard_array.sum() / scorecard_array.size)
