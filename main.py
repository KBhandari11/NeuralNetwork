import numpy as np
from neuralNetworks import neuralNetwork
from mat4py import loadmat
from scipy.optimize import minimize

data = loadmat('ex4data1.mat')
X = data.get('X')
y= data.get('y')
X =np.array(X);
Y =np.array(y);
m = X.shape[0]

# Set the hyperparameters
n_x = 400    #No. of neurons in first layer
n_h = 25     #No. of neurons in hidden layer
n_y = 10    #No. of neurons in output layer
num_of_iters = 1
#beginning of experiment using the preexisting weights
data = loadmat('ex4weights.mat')
W1 = data.get('Theta1')
W2 = data.get('Theta2')
W1 = np.array(W1);
W2 = np.array(W2);
parameters={
    "W1": W1,
    "W2": W2
}
#with lamda value 0; cost should be 0.287629
learning_rate = 0
num_of_iters = 1
nN = neuralNetwork(X, Y, n_x, n_h, n_y, learning_rate)
cost, grad = nN.model(parameters)
print(cost)
#using the lamda value 1 the cost should be 0.383770
learning_rate = 1
nN = neuralNetwork(X, Y, n_x, n_h, n_y, learning_rate)
cost, grad = nN.model(parameters)
print(cost)
#using the lamda value 3 the cost should be 0.576051
learning_rate = 3
nN = neuralNetwork(X, Y, n_x, n_h, n_y, learning_rate)
cost, grad= nN.model(parameters)
print(cost)
#using a random variable
#<---------------------->
learning_rate = 1
num_of_iters = 100
nN = neuralNetwork(X, Y, n_x, n_h, n_y, learning_rate)
parameters = neuralNetwork.initialize_parameters(n_x, n_h, n_y)
print (minimize(nN.model, parameters, options={
    'maxiter': 100,
    'maxfun': 500
}))
#<------------------------>

#X_test = np.array([[1], [1],[1]])

#y_predict = nN.predict(X_test, trained_parameters)

#print('Neural Network prediction for example ({:d}, {:d}, {:d}) is {:d}'.format(X_test[0][0], X_test[1][0],X_test[2][0], y_predict))

