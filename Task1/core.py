import numpy as np
import matplotlib.pyplot as plt
#defining the XOR logic rn
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

#lets define the number of required neurons
input_neurons= 2
hidden_neurons = 2
output_neurons = 1
#here we are defining how are the weights choosen: randomly between 0 and 1 with every number with an equal probability of occurance.
W1=np.random.uniform(size=(input_neurons,hidden_neurons))
W2 = np.random.uniform(size = (hidden_neurons,output_neurons))

#each neuron in the hidden layer gets it's own bias
B1 = np.zeros((1,hidden_neurons)) #(here B1 and B2 are the bias for the particular case)
#(1 represents that the bias is a 2D row matrix and not simply a list)
#(np.zeros tells the numpy to create an array filled with 0.0)
#(since hidden layer has 2 neurons the b1 would look like (a 2D matrix) [[0.0 0.0]] i.e, the each neuron gets its own bias)
#if there are more than one input rows of X , (above we have 4) but only 1 for bias i.e, [[0.0 0.0]] now that will be extended for all 4 as [[0.0 0.0],[0.0 0.0],[0.0 0.0],[0.0 0.0]]
B2 = np.zeros((1,output_neurons))

#The dimensions of these matrices are strict. W1 must be (input x hidden) because that is the "pathway" the data travels.


#now we use the "sigmoid function inorder to squish the output to between the range 0 and 1"
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

#multiplying weights by the inputs and adding the bias
#the below code represents the data passing between the layers(forward passing)
#np.dot is used for the matrix multiplication


learning_rate = 0.1
epochs = 200
losses = []

for epoch in range(epochs):

    hidden_layer_input = np.dot(X,W1) + B1
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output,W2) + B2
    predicted_output = sigmoid(output_layer_input)

    loss = np.mean(np.square(Y - predicted_output))
    losses.append(loss)

    error = Y - predicted_output
    d_predicted_output  = error * sigmoid_derivative(predicted_output) # this determines how much the the ouput must be changed the next time

    #also now inorder to check whether the error was from the inner hidden layer we check
    error_hidden_layer = d_predicted_output.dot(W2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    #updating the weights: this is where the actual learning happens
    W2 +=hidden_layer_output.T.dot(d_predicted_output)*learning_rate
    W1 +=X.T.dot(d_hidden_layer)*learning_rate
    # Updating biases (important for the model's flexibility)
    B2 += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    B1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if epoch % 40 == 0:
        print(f"Epoch  {epoch}, Loss {loss}")


# Check final predictions
print("\nFinal Predictions after 200 epochs:")
print(predicted_output)

# Plot the loss graph as required by the task
plt.plot(losses)
plt.title("Loss for 200 Epochs (XOR Task)")
plt.xlabel("Epochs")
plt.ylabel("Loss Score")
plt.show()
