import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

iris = load_iris()
X = iris.data 
X = (X - X.mean()) / X.std()

encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(iris.target.reshape(-1, 1))



input_neurons = 4
hidden_neurons = 5
output_neurons = 3

learning_rate = 0.1

w1 = np.random.uniform(size = (input_neurons,hidden_neurons))
w2 = np.random.uniform(size = (hidden_neurons,output_neurons))

b1 = np.zeros((1,hidden_neurons))
b2 = np.zeros((1,output_neurons))

def train_model(activation_func, derivative_function, epochs=200):
    global b1, b2
    w1 = np.random.uniform(size=(input_neurons, hidden_neurons)) # (4, 5)
    w2 = np.random.uniform(size=(hidden_neurons, output_neurons)) # (5, 3)
    
    # Resetting biases inside
    b1_inner = np.zeros((1, hidden_neurons))
    b2_inner = np.zeros((1, output_neurons))

    losses = []

    for epoch in range(epochs):
        hli = np.dot(X, w1) + b1_inner
        hlo = activation_func(hli)

        oli = np.dot(hlo, w2) + b2_inner
        predicted_output = activation_func(oli)

        loss = np.mean(0.5 * np.square(Y - predicted_output))
        losses.append(loss)

        error = Y - predicted_output
        d_predicted_output = error * derivative_function(predicted_output)
        ehl = d_predicted_output.dot(w2.T)
        d_hidden_layer = ehl * derivative_function(hlo)

        w2 += hlo.T.dot(d_predicted_output)*learning_rate
        w1 += X.T.dot(d_hidden_layer)*learning_rate

        b2 += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

        if epoch % 40 == 0:
            print(f"Epoch  {epoch}, Loss {loss}")


    return losses



def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)

def bumpy_relu(x):
    return np.where(x>0,x+0.5*np.sin(x),0) #return 0 if x<0 & if x>0 returns (x+0.5*sin(x))
def bumpy_relu_derivative(x):
    return np.where(x>0,1+0.5*np.cos(x),0) # return 0 if x<0 but, if x>0 gives (1+0.5*cos(x))


    

sigmoid_history = train_model(sigmoid, sigmoid_derivative)
bumpy_history = train_model(bumpy_relu, bumpy_relu_derivative)

# Plot comparison
plt.plot(sigmoid_history, label='Standard Sigmoid')
plt.plot(bumpy_history, label='Custom Bumpy-ReLU')
plt.legend()
plt.title("Performance Comparison")
plt.show()




"""magine you want to test two different types of running shoes (these are your Activation Functions: Sigmoid vs. Bumpy-ReLU) to see which one helps a runner go faster. 

Outside the function: Setting W1 and W2 outside is like marking the Starting Line.

Inside the function: If you don't reset the weights inside the function, the second shoe you test would start from where the first shoe finished!

By resetting W1 and W2 inside the function, we ensure every "test run" starts from the exact same random starting point. This makes the comparison fair"""