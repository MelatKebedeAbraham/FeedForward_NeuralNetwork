import numpy as np
#import sys
 
#path = 'C:/Users/Dell-PC/Desktop/ML Assignment/Output.txt'
#sys.stdout = open(path, 'w')

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivative(x):
     return sigmoid(x)*(1.0-sigmoid(x))

def feedforward(inputs, weight_one, weight_two, desired_output):
        input_layer_one = []
        layer_one = sigmoid(np.dot(inputs,weight_one))
        input_layer_one.append(1)
        input_layer_one.append(layer_one)
        output_layer =sigmoid(np.dot(input_layer_one, weight_two))
        #print(output_layer)
        output = (np.square(np.subtract(output_layer,desired_output)))/2
        #print(output)
        Error = np.sum(output)/2
        #print(Error)
        return Error

def full_feedforward(assigned_input):
    inputs = assigned_input[0]
    weight_one = assigned_input[1]
    weight_two = assigned_input[2]
    desired_output = assigned_input[3]
    return feedforward(inputs, weight_one, weight_two, desired_output)

def return_feedforward(inputs, weight_one, weight_two):
    input_layer_one = []
    layer_one = sigmoid(np.dot(inputs,weight_one))
    input_layer_one.append(1)
    input_layer_one.append(layer_one)
    output_layer =sigmoid(np.dot(input_layer_one, weight_two))
    #print(output_layer)
    output_layer_average_value = np.sum(output_layer)/2
    return layer_one, output_layer_average_value

def full_return_feedforward(assigned_input):
    inputs = assigned_input[0]
    weight_one = assigned_input[1]
    weight_two = assigned_input[2]
    return return_feedforward(inputs, weight_one, weight_two)

def backpropagation(inputs,weight_one, weight_two,layer_one,Total_Error,output_layer_average_value):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weight_two = np.dot(layer_one.T, (0.65*(Total_Error) * sigmoid_derivative(output_layer_average_value)))
        # update the weights with the derivative (slope) of the loss function
        d_weight_one = np.dot(np.array(inputs).T,  (np.dot(0.65*(Total_Error) * sigmoid_derivative(output_layer_average_value), d_weight_two.T) * sigmoid_derivative(layer_one)))
        print("This is weight One:",weight_one)
        print("This is weight Two:",weight_two)
        weight_one -= d_weight_one
        weight_two -= d_weight_two
        print("This is Changed weight One:",weight_one)
        print("This is Changed weight Two:",weight_two)
        return weight_one, weight_two
    
def full_backpropagation(assigned_input):
    inputs = assigned_input[0]
    weight_one = assigned_input[1]
    weight_two = assigned_input[2]
    layer_one = assigned_input[3][0]
    Total_Error = assigned_input[4]
    output_layer_average_value = assigned_input[3][1]
    return backpropagation(inputs,weight_one, weight_two,layer_one,Total_Error,output_layer_average_value)
    
    
input_one = [1, 0.2]
input_two = [1, 0.6]

all_errors = []
iteration_number = []

items = [input_one,input_two]
full_input = [[0.2,0.3],[[0.1,0.8],[0.3,0.2]],[0.4,0.2]]

print("                                        Start Iteration                                      ")

for i in range(200):
     print("-------------------------------------------------------------------------------------------------------------------------")
     print("-------------------------------------------------------------------------------------------------------------------------")
     print("This is Iteration :",i)
     iteration_number.append(i)
     total_error = 0
     for i in items:
        full_input.insert(0, i)
        print("items:",full_input)
        total_error += full_feedforward(full_input)
        full_input.remove(i)
     if total_error == 0.000001:
         print("The training is sucessful at Iteration number=",i)
     all_errors.append(total_error)      
     print("This is the Error:", total_error)
     print("-------------------------------------------------------------------------------------------------------------------------")
     if total_error >= 0.000001:
         for i in items:
            Desired = full_input.pop(-1)
            full_input.insert(0,i)
            print(full_input)
            # value = (layer_one, output_layer_average_value)
            value = full_return_feedforward(full_input)
            full_input.append(value)
            full_input.append(total_error)
            change_weight = full_backpropagation(full_input)
            full_input.pop(-1)
            full_input.pop(3)
            full_input.remove(i)
            #print("Change Weight:",change_weight)
            full_input[0][0] = change_weight[0][0]
            full_input[0][1] = change_weight[0][1]
            full_input[1][0][0] = change_weight[1][0][0]
            full_input[1][0][1] = change_weight[1][0][1]
            full_input[1][1][0] = change_weight[1][1][0]
            full_input[1][1][1] = change_weight[1][1][1]
            full_input.append(Desired)

print("-------------------------------------------------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------------------------------------------------")                         
print("List of the Error Vector",all_errors)
print("List of the Iteration Vector",iteration_number)

print("-------------------------------------------------------------------------------------------------------------------------")
print("#########################################################################################################################")
# importing the required module 
import matplotlib.pyplot as plt 
  
# plotting the points  
plt.plot(iteration_number,all_errors) 
    
# naming the x axis 
plt.xlabel('x - axis: Iteration') 
# naming the y axis 
plt.ylabel('y - axis: Error') 
    
# giving a title to my graph 
plt.title('The Graph for the Error Iteration Graph!') 
    
# function to show the plot 

plt.show()

